# Tanbr Setup and Usage

This repository contains the code and instructions for training and testing Tanbr methods in the T5-GLUE experiments.


# Installation

To install the required packages, run the following command:

<pre lang="markdown">
python setup.py develop
</pre>

Make sure to have the packages version same as the ones specified in requirements.txt. The above command will ensure that.

# Temporary JSON Files

Before running the code, create a directory named temp_jsons. This directory will be used to save temporary JSON files during the execution of the code.

# Fine-tuning and Inference

The repository provides different commands for training and testing the models with Tanbr. To ensure the validity of the hyperparameters we used, execute the following commands:

Fine-tuning

<pre lang="markdown">
python ./finetune_t5_trainer.py configs/adapter/ht_nucb_routing.json
</pre>

Inference


<pre lang="markdown">
python ./finetune_t5_trainer.py configs/adapter/ht_nucb_routing.json -k do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
</pre>
