import argparse
import os
import yaml

import wandb

from gnd_dataset import GNDDataset
from trainer import Trainer
from utils import init_prompt_model


parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--data_dir", type=str, help="Path to the GND dataset directory.")
parser.add_argument("--gnd_graph", type=str, help="Path to the GND graph file (pickle).")
parser.add_argument("--config", type=str, help="Path to the configuration file.")

arguments = parser.parse_args()
data_dir = arguments.data_dir
gnd_graph = arguments.gnd_graph
config_path = arguments.config

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

model_name = config["model_name"]

model, tokenizer = init_prompt_model(
    model_name=model_name,
    prompt_config=config["prompt_config"]
)

# Load GND dataset
gnd_ds = GNDDataset(
    data_dir=data_dir,
    gnd_graph=gnd_graph,
    config=config
)

# Tokenize the datasets
gnd_ds.tokenize_datasets(tokenizer=tokenizer)

# Split the dataset into train, validation, and test sets
train_ds = gnd_ds["train"]
valid_ds = gnd_ds["validate"]
test_ds = gnd_ds["test"]

# How many parameters are in the model?
total_model_params = 0
num_trained_params = 0
for p in model.parameters():
    if p.requires_grad:
        num_trained_params += p.numel()
    else:
        total_model_params += p.numel()
print("Total Model Parameters: {}, Trainable Parameters: {}, Percentage {}".format(
    total_model_params, 
    num_trained_params,
    num_trained_params / (total_model_params + num_trained_params)
    )
    )
prompt_type = config["prompt_config"]["type"]
layer = config["prompt_config"]["at_layer"]
num_tokens = config["prompt_config"]["num_prompt_tokens"]

wandb.init(
      # Set the project where this run will be logged
      project="xmlc-knowledge",
      name=f"{model_name}-{prompt_type}-l-{layer}-n-{num_tokens}",
      # Track hyperparameters and run metadata
      config={
          "model_name": model_name,
          "config": config,
          "n_train": len(train_ds),
          "params": num_trained_params,
      })

trainer = Trainer(config)
output_dir = config["checkpoint_path"]

if output_dir not in os.listdir():
    os.mkdir(output_dir)
trainer.train(model, train_ds, valid_ds, output_dir=output_dir)
    