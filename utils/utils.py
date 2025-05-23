import glob
import os
from dataclasses import asdict
from logging import getLogger
from hyperformer.third_party.utils import (
    assert_all_frozen,
    freeze_embeds,
    freeze_params,
    save_json)
from transformers.models.t5.modeling_t5 import T5LayerNorm

from hyperformer.adapters import (MetaAdapterController, 
                              AdapterLayersHyperNetController, AdapterLayersOneHyperNetController)
# from hyperformer.adapters import AdapterController
from hyperformer.adapters.adapter_controller_fast import AdapterController
from hyperformer.data import TASK_MAPPING
import pdb

logger = getLogger(__name__)


def create_dir(output_dir):
    """
    Checks whether to the output_dir already exists and creates it if not.
    Args:
      output_dir: path to the output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def handle_metrics(split, metrics, output_dir): #, gcs_bucket=None):
    """
    Prints and saves metrics or a general dictionary of results.

    Args:
        split: one of train, val, test, or training arguments.
        metrics: metrics dict
        output_dir: where to save the metrics, if gcs_bucket is given
        we save the results also on the given bucket.
    """
    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json_file(metrics, f"{split}_results.json", output_dir)


def save_json_file(json_dict, outfile_name, output_dir):
    """
    Saves the given dictionary as a json file to output_dir and also
    the given bucket if given.
    """
    save_json(json_dict, os.path.join(output_dir, outfile_name))


def get_training_args(arguments_list):
    """
    Concatenate all training arguments except evaluation strategy which
    is not Json serializable.
    Args:
        arguments_list: list of dataclasses.
    Return:
        arguments: concatenated arguments.
    """
    all_arguments = {}
    for arguments in arguments_list:
        all_arguments.update(asdict(arguments))
    all_arguments.pop("evaluation_strategy")
    return all_arguments


def get_last_checkpoint_path(output_dir):
    """
    Finds the path for the last checkpoint saved in the output_dir
    Args:
        output_dir:  output_dir
    Returns:
        path to the last checkpoint saved in the output dir.
    """
    paths = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if len(paths) == 0:
        return output_dir
    else:
        checkpoints = [int(checkpoint.split('-')[-1]) for checkpoint in paths]
        max_checkpoint = max(checkpoints)
        return os.path.join(output_dir, "checkpoint-" + str(max_checkpoint))


def use_task_specific_params(model, task):
    """Update config with task specific params during evaluation."""
    task_dataset = TASK_MAPPING[task]
    task_specific_config = task_dataset.task_specific_config
    if task_specific_config is not None:
        logger.info(f"using task specific params for {task}: {task_specific_config}")
        model.config.update(task_specific_config)


def reset_config(model, config):
    """Resets the config file to the one provided."""
    model.config = config
    logger.info(f"config is reset to the initial values.")


def freezing_params(model, training_args, model_args, adapter_args):
    """
    Freezes the model parameters based on the given setting in the arguments.
    Args:
      model: the given model.
      training_args: defines the training arguments.
      model_args: defines the model arguments.
      adapter_args: defines the adapters arguments.
    """
    # If we are training adapters, we freeze all parameters except the
    # parameters of computing task embeddings and adapter controllers.
    trainable_params = []
    non_router_params = []
    # True
    if training_args.train_adapters or adapter_args.train_lora or adapter_args.train_ia3:
        freeze_params(model)
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (AdapterController)):
                for param_name, param in sub_module.named_parameters():
                    #pdb.set_trace()
                    if adapter_args.only_train_router:
                        if ('router' in name+param_name) or ('baseline' in name+param_name) or ('decoder' in name+param_name):
                            trainable_params.append(name + '.' + param_name)
                            param.requires_grad = True
                    else:
                        trainable_params.append(name + '.' + param_name)
                        param.requires_grad = True
                        if 'router' not in (name + '.' + param_name):
                            non_router_params.append(param)    
    # False
    if model_args.freeze_model:
        freeze_params(model)

    # False
    # Freezes all models parameters except last linear layer of decoder.
    if model_args.freeze_model_but_lm_head:
        freeze_params(model)
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # False
    if model_args.freeze_embeds:
        freeze_embeds(model)

    # False
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    # False
    # In case of meta-adapters and if task-embeddings are paramteric,
    # freezes all parameters except task-embedding parameters.
    if model_args.freeze_model_but_task_embeddings:
        freeze_params(model)
        if adapter_args.adapter_config_name == "meta-adapter" and \
                not isinstance(model.task_embedding_controller.task_to_embeddings, dict):
            for param in model.task_embedding_controller.task_to_embeddings.parameters():
                param.requires_grad = True

    # False
    # Unfreezes last linear layer of decoder.
    if model_args.unfreeze_lm_head:
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # True
    # Unfreezes layer norms.
    if model_args.unfreeze_layer_norms:
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, T5LayerNorm):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
                    trainable_params.append(name + '.' + param_name)
                    if 'router' not in (name + '.' + param_name):
                        non_router_params.append(param)

    # False
    if model_args.unfreeze_model:
        for param in model.parameters():
            param.requires_grad = True
    #pdb.set_trace()
    print('The trainable params are ', trainable_params)
    print(f'Total trainable params are {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'Total non router trainable parameters are {sum(p.numel() for p in non_router_params)}')
