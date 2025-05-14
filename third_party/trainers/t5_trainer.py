"""Implements a T5 trainer class doing training and evaluation."""
import pickle
import collections
from collections import Counter
import math
import time
import numpy as np
import os
import torch
from packaging import version
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch import nn
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedModel, logging
from transformers import Trainer
from transformers.models.fsmt import FSMTConfig
from transformers.utils import (is_torch_tpu_available)
from transformers.integrations import (hp_params)
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import (TrainOutput)
from transformers.trainer_utils import (set_seed)

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

from typing import Any, Dict, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset

from hyperformer.adapters import MetaAdapterConfig
from hyperformer.utils import use_task_specific_params, reset_config
from hyperformer.data import MultiTaskBatchSampler, MultiTaskTempBatchSampler

logger = logging.get_logger(__name__)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

def get_global_step_sum(global_step):
    #tensor = torch.tensor(global_step, dtype=torch.long)
    tensor = torch.tensor(global_step, dtype=torch.int32).to(torch.device(f"cuda:{dist.get_rank()}"))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


class T5Trainer(Trainer):
    def __init__(self, config=None, data_args=None, dataset_sizes=None, adapter_config=None,
                 multi_task_compute_metrics=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if config is None:
            assert isinstance(
                self.model, PreTrainedModel
            ), f"If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is {self.model.__class__}"
            self.config = self._actual_model(self.model).config
        else:
            self.config = config

        self.adapter_config = adapter_config
        self.multi_task_compute_metrics = multi_task_compute_metrics
        self.dataset_sizes = dataset_sizes
        self.data_args = data_args
        self.vocab_size = self.config.tgt_vocab_size if isinstance(self.config, FSMTConfig) else self.config.vocab_size

        if self.args.label_smoothing != 0 or (self.data_args is not None and self.data_args.ignore_pad_token_for_loss):
            assert (
                    self.config.pad_token_id is not None
            ), "Make sure that `config.pad_token_id` is correcly defined when ignoring `pad_token` for loss calculation or doing label smoothing."

        if self.config.pad_token_id is None and self.config.eos_token_id is not None:
            logger.warn(
                f"The `config.pad_token_id` is `None`. Using `config.eos_token_id` = {self.config.eos_token_id} for padding.."
            )
        self.kl_loss = nn.KLDivLoss(reduction="none", log_target=True)
        if self.args.label_smoothing == 0:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        else:
            # dynamically import label_smoothed_nll_loss
            from hyperformer.third_party.utils import label_smoothed_nll_loss

            self.loss_fn = label_smoothed_nll_loss
        self.args.save_safetensors = False

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use
        something else, you can pass a tuple in the Trainer's init through
        :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            if self.adapter_config.routing_estimator == "skill_routing":
                skill_param = [param for param_name, param in self.model.named_parameters() if "skill_weights" in param_name]
                other_param = [param for param_name, param in self.model.named_parameters() if "skill_weights" not in param_name]
                optimizer_grouped_parameters = [{"params": skill_param, "lr": self.args.learning_rate * self.adapter_config.skill_lr_ratio}, {"params": other_param, "lr": self.args.learning_rate}]
            else:
                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
            if self.args.adafactor:
                self.optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    scale_parameter=False,
                    relative_step=False,
                )

            else:
                self.optimizer = AdamW(
                    optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon
                )

        if self.lr_scheduler is None:
            self.lr_scheduler = self._get_lr_scheduler(num_training_steps)
        else:  # ignoring --lr_scheduler
            logger.warn("scheduler is passed to `Seq2SeqTrainer`, `--lr_scheduler` arg is ignored.")


    def _get_lr_scheduler(self, num_training_steps):
        schedule_func = arg_to_scheduler[self.args.lr_scheduler]
        if self.args.lr_scheduler == "constant":
            scheduler = schedule_func(self.optimizer)
        elif self.args.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(self.optimizer, num_warmup_steps=self.args.warmup_steps)
        else:
            scheduler = schedule_func(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
        return scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if is_torch_tpu_available() and xm.xrt_world_size() > 1:
            num_replicas = xm.xrt_world_size()
            rank = xm.get_ordinal()
        elif self.args.local_rank != -1:
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            num_replicas = 1
            rank = 0
        if self.args.temperature > 1:
            print(f"Using temperature sampler")
            return MultiTaskTempBatchSampler(self.dataset_sizes, self.args.train_batch_size,
                                        self.args.temperature, rank=rank,
                                        num_replicas=num_replicas)
        else:
            return MultiTaskBatchSampler(self.dataset_sizes, self.args.train_batch_size,
                                        self.args.temperature, rank=rank,
                                        num_replicas=num_replicas)

    def _compute_loss(self, model, inputs, labels):
        if self.args.label_smoothing == 0:
            if self.data_args is not None and self.data_args.ignore_pad_token_for_loss:
                # force training to ignore pad token
                logits = model(**inputs, use_cache=False)[0]
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
                
            else:
                # compute usual loss via models
                loss, logits = model(**inputs, labels=labels, use_cache=False)[:2]
        elif self.adapter_config.routing_estimator == 'ht_nucb_routing':
            
            tasks = inputs["task"]
            config_tasks = self.multi_task_compute_metrics.keys()
            task_counts = Counter(tasks)
            weight = [task_counts.get(task_name, 0) / len(tasks) for task_name in config_tasks]
            if weight not in self.weights:
                self.weights.append(weight)
            expert_probs = model.module.get_expert_prob(weight)
            
            output, extra_params = model(**inputs,return_dict=False, use_cache=False, weight=expert_probs)

            logits = output[0]

            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss, _, batched_loss = self.loss_fn(lprobs, labels, self.args.label_smoothing, ignore_index=self.config.pad_token_id)
    
            
            loss_per_task = {task: [] for task in config_tasks}
            for l, task in zip(batched_loss, tasks):
                if task in loss_per_task:
                    loss_per_task[task].append(l)
            average_loss_per_task = [sum(losses) / len(losses) if losses else 0 for task, losses in loss_per_task.items()]
            average_loss_per_task = torch.tensor(average_loss_per_task).cpu().numpy()
            model.module.router_update(average_loss_per_task)
            
            
            
        else:
            if self.config.train_adapters or self.adapter_config.train_lora or self.adapter_config.train_ia3:
                output, extra_params = model(**inputs,return_dict=False, use_cache=False)
                #print(f"output type: {type(output)}\n output:{output}")
                #print(f"type: {type(extra_params)}\n output:{extra_params}")
                logits = output[0]
                #print(f"output[0]_logits type: {type(logits)}\n logits:{logits}")
                if self.adapter_config.routing_estimator == 'reinf_bl_routing':
                    adapter_probs_list = extra_params[0]
                    baselines_list = extra_params[1]
                    samples_list = extra_params[2]
                    load_loss = extra_params[3]
                    supervised_loss = extra_params[4]
                else:
                    load_loss = extra_params[0]
                    supervised_loss = extra_params[1]
            else:
                # compute label smoothed loss
                logits = model(**inputs, use_cache=False)[0]
                #print(f"[0]_logits type: {type(logits)}\n logits:{logits}")
            if self.adapter_config.probe_input_features:
                loss = torch.tensor(0).to(labels.device)
            else:
                #print(f"logits type: {type(logits)}\n logits:{logits}")
                lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                loss, _, batched_loss = self.loss_fn(lprobs, labels, self.args.label_smoothing, ignore_index=self.config.pad_token_id)

            #print("loss!!!",loss)
            if (self.config.train_adapters or self.adapter_config.train_lora or self.adapter_config.train_ia3) and self.model.training:
                if self.adapter_config.routing_estimator == 'reinf_bl_routing':
                    #bs, n_layers, num_adapters
                    adapter_probs = torch.cat([probs.unsqueeze(1) for probs in adapter_probs_list], dim=1)
                    #bs, n_layers, num_adapters
                    samples = torch.cat([sample.unsqueeze(1) for sample in samples_list], dim=1)
                    #bs, n_layers
                    baseline_vals = torch.cat([baseline_val for baseline_val in baselines_list], dim=1)
                    #bs, n_layers
                    batched_loss = batched_loss.repeat(1, len(baselines_list))
                    log_adapter_probs = torch.log(adapter_probs + 1e-20)
                    advantage_vals = batched_loss.detach() - baseline_vals
                    #bs, n_layers, num_adapters
                    entropy_loss = - (log_adapter_probs)*adapter_probs
                    entropy_loss = torch.mean(torch.sum(entropy_loss, dim=2))
                    #bs, n_layers
                    policy_loss = torch.sum(log_adapter_probs*samples, dim=-1) * advantage_vals.detach()
                    policy_loss = torch.mean(policy_loss)
                    delta = 1.0
                    abs_advantage_vals = torch.abs(advantage_vals)
                    with torch.no_grad():
                        mask1 = (abs_advantage_vals < delta).float()
                        mask2 = (abs_advantage_vals >= delta).float()
                    value_loss1 = 0.5 * (advantage_vals)**2
                    value_loss2 = delta * (abs_advantage_vals - 0.5 * delta)
                    value_loss = value_loss1 * mask1 + value_loss2 * mask2
                    value_loss = torch.mean(value_loss)

                    loss = loss + self.adapter_config.policy_weight * policy_loss + \
                        self.adapter_config.policy_entropy_weight * entropy_loss + \
                            self.adapter_config.value_function_weight * value_loss + \
                                self.adapter_config.load_loss_weight * load_loss + \
                                    self.adapter_config.supervised_loss_weight * supervised_loss
                    if (self.state.global_step + 1) % self.args.logging_steps == 0:
                        self.log({'policy_loss': self.adapter_config.policy_weight * torch.sum(policy_loss).item()})
                        self.log({'entropy_loss': self.adapter_config.policy_entropy_weight * torch.sum(entropy_loss).item()})
                        self.log({'value_loss': self.adapter_config.value_function_weight * torch.sum(value_loss).item()})
                        self.log({'load_loss': self.adapter_config.load_loss_weight * torch.sum(load_loss).item()})
                        self.log({'supervised_loss': self.adapter_config.supervised_loss_weight * torch.sum(supervised_loss).item()})
                elif self.adapter_config.routing_estimator == "adamix_routing":
                    output_2, extra_params_2 = model(**inputs, use_cache=False)
                    logits_2 = output_2[0]
                    lprobs_2 = torch.nn.functional.log_softmax(logits_2, dim=-1)
                    consistency_loss = 0.5 * (self.kl_loss(lprobs, lprobs_2) + self.kl_loss(lprobs_2, lprobs))
                    consistency_loss = torch.mean(torch.sum(consistency_loss, dim=-1))
                    if (self.state.global_step + 1) % self.args.logging_steps == 0:
                        self.log({'load_loss': self.adapter_config.load_loss_weight * torch.sum(load_loss).item()})
                        self.log({'supervised_loss': self.adapter_config.supervised_loss_weight * torch.sum(supervised_loss).item()})
                        self.log({'consistency_loss': consistency_loss.item()})
                    loss = loss + self.adapter_config.load_loss_weight*load_loss + self.adapter_config.supervised_loss_weight * supervised_loss + consistency_loss
                else:
                    if (self.state.global_step + 1) % self.args.logging_steps == 0:
                        self.log({'load_loss': self.adapter_config.load_loss_weight * torch.sum(load_loss).item()})
                        self.log({'supervised_loss': self.adapter_config.supervised_loss_weight * torch.sum(supervised_loss).item()})     
                    loss = loss + self.adapter_config.load_loss_weight*load_loss + self.adapter_config.supervised_loss_weight * supervised_loss

        return loss, logits

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        multitask_sampler = self._get_train_sampler()
        return DataLoader(self.train_dataset, batch_sampler=multitask_sampler,
                          collate_fn=self.data_collator, num_workers=40, pin_memory=False)

    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        loss, _, = self._compute_loss(model, inputs, labels)
        # print("loss!!",loss)
        return loss

    def evaluate(self, eval_datasets: Optional[Dict[str, Dataset]] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        results = {}
        if eval_datasets is None:
            eval_datasets = self.eval_dataset



        for eval_task, eval_dataset in eval_datasets.items():
            self.compute_metrics = self.multi_task_compute_metrics[eval_task]
            model_config = self.model.config

            use_task_specific_params(self.model, eval_task)

            if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
                raise ValueError("eval_dataset must implement __len__")



            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            eval_dataloader.batch_size = 128
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None
            )
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

            tasks_metric = {eval_task + "_" + k: v for k, v in output.metrics.items()}
            for key in sorted(tasks_metric.keys()):
                logger.info(f"  {key} = {tasks_metric[key]}")
            results.update(tasks_metric)
            reset_config(self.model, model_config)

        # Computes the average metrics across all the tasks without their corresponding losses.
        metrics = [results[key] for key in results.keys() if "loss" not in key]
        results['eval_average_metrics'] = np.mean(metrics)
        self.log(results)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, results)
        if self.adapter_config.probe_input_features:
            acc = round (self.adapter_config.num_count_task_pred/ self.adapter_config.den_count_task_pred, 4)
            print(f'Overall accuracy is {acc}')

        return results



        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)


        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach(), loss_avg

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            print('Should not enter here')
            # Seed must be set before instantiating the model when using model_init.
            # set_seed(self.args.seed)
            set_seed(self.data_args.data_seed)

            model = self.call_model_init(trial)

            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        # Mixed precision training with apex (torch < 1.6)
        model = self.model
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)


        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )
        # find_unused_parameters breaks checkpointing as per
        # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", max_steps)

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", self.state.global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(self.args.device)
        self._logging_loss_scalar = 0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        global_step = 0

        loss_log = []

        self.weights = []
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and (
                    isinstance(train_dataloader.sampler,DistributedSampler)
                    or isinstance(train_dataloader.batch_sampler, MultiTaskBatchSampler)):
                if isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(epoch)
                else:
                    train_dataloader.batch_sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                
                #print(f"P_inputs type: {type(inputs)}\n Inputs:{inputs}")
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                # update adapter temperature
                if self.config.train_adapters or self.adapter_config.train_lora or self.adapter_config.train_ia3:
                    if self.adapter_config.routing_estimator == 'gs_st_routing':
                        if (self.state.global_step + 1) % self.args.eval_steps == 0:
                            temp = self.adapter_config.adapter_temp
                            temp = np.maximum(temp*np.exp(-self.adapter_config.anneal_rate*self.state.global_step), self.adapter_config.min_temp)
                            self.adapter_config.adapter_temp = temp
                            # logger.info(f"adapter temperature value is {self.adapter_config.adapter_temp}")
                            self.log({'adapter_temp': temp})
                if self.adapter_config.same_init_then_branch > 0:
                    if self.state.global_step > self.adapter_config.same_init_then_branch:
                        self.adapter_config.same_init_then_branch = -1

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if (
                        ((step + 1) % self.args.gradient_accumulation_steps != 0)
                        and self.args.local_rank != -1
                        and _use_ddp_no_sync
                ):
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)
                    tr_loss += tr_loss_step
                self._total_flos += self.floating_point_ops(inputs)
                loss_log.append(tr_loss.detach().cpu().item())

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= self.args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                    #self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                global_step += 1
                total_global_step = get_global_step_sum(global_step)
                if self.control.should_epoch_stop or self.control.should_training_stop or total_global_step >= max_steps:
                    break
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            #self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)
            print("epoch:{}, loss: {}".format(epoch, tr_loss))
            '''
            if epoch % 10==0:
                loss_save_dir = os.path.join(self.args.output_dir, 'loss_data.pkl')
                with open(loss_save_dir, "wb") as f:
                    pickle.dump(loss_log, f)
        
                last_checkpoint_dir = os.path.join(self.args.output_dir, 'last_checkpoint')
                os.makedirs(last_checkpoint_dir, exist_ok=True)
                self._save(last_checkpoint_dir)
                torch.save(self.optimizer.state_dict(), os.path.join(last_checkpoint_dir, 'optimizer.pt'))
                torch.save(self.lr_scheduler.state_dict(), os.path.join(last_checkpoint_dir, 'scheduler.pt'))
                self.state.save_to_json(os.path.join(last_checkpoint_dir, 'trainer_state.json'))
            '''
            
            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop or total_global_step >= max_steps:
                break
        loss_save_dir = os.path.join(self.args.output_dir, 'loss_data.pkl')
        with open(loss_save_dir, "wb") as f:
            pickle.dump(loss_log, f)
        
        last_checkpoint_dir = os.path.join(self.args.output_dir, 'last_checkpoint')
        os.makedirs(last_checkpoint_dir, exist_ok=True)
        self._save(last_checkpoint_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(last_checkpoint_dir, 'optimizer.pt'))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(last_checkpoint_dir, 'scheduler.pt'))
        self.state.save_to_json(os.path.join(last_checkpoint_dir, 'trainer_state.json'))

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        """
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(model, PreTrainedModel):
                self.model = model.from_pretrained(self.state.best_model_checkpoint, adapter_config=self.adapter_config)
                self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)
        """

        if self._total_flos is not None:
            self.store_flos()
            self.log({"total_flos": self.state.total_flos})

        
        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        return TrainOutput(self.state.global_step, tr_loss.item() / self.state.global_step, {})

    def prediction_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model.
                Most models expect the targets under the argument :obj:`labels`.
                Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        # print("Entered here")
        # print(f"task is {inputs['task']}")
        # print(self.adapter_config.analysis_list)
        self.adapter_config.analysis_list = {}
        # print(f'length of complete analysis list is {len(self.adapter_config.complete_analysis_list)}')
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": self.config.max_length,
            "num_beams": self.config.num_beams
        }
        gen_kwargs["task"] = inputs["task"]
        gen_kwargs['orig_task'] = inputs['orig_task']
        gen_kwargs['hash_lbl'] = inputs['hash_lbl']
        gen_kwargs["task_embedding"] = model.task_embedding_controller(inputs["task"]) if \
            (self.config.train_adapters and isinstance(self.adapter_config, MetaAdapterConfig)) else None
        if self.args.predict_with_generate and not self.args.prediction_loss_only:
            generated_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        end_time = time.time()
        self.adapter_config.eval_time += (end_time - start_time)
        labels = inputs.pop("labels")
        loss = torch.tensor(0).to(labels.device)
        if self.args.prediction_loss_only:
            return (loss, None, None)

        logits = generated_tokens if self.args.predict_with_generate else logits

        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        return (loss, logits, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id`"
                f" is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
