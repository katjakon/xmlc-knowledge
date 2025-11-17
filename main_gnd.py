import argparse
import os
import pickle
import yaml

import torch
import wandb
from datasets import Dataset

from gnd_graph import GNDGraph
from data_collator import GraphDataCollator
from trainer import Trainer
from utils import init_prompt_model, load_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--dev", action="store_true", help="Run in development mode with a smaller dataset.")
arguments = parser.parse_args()
config_path = arguments.config
dev = arguments.dev
# Load config 
config = load_config(config_path)

exp_name = config["experiment_name"]
output_dir = config["checkpoint_path"]
output_dir = os.path.join(output_dir, exp_name)

if os.path.exists(output_dir):
    print(f"Output directory {output_dir} already exists. Please remove it or choose a different name.")
    exit(1)

os.makedirs(output_dir)

# Load GND graph
gnd_path  = config["graph_path"]
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

model_name = config["model_name"]

model, tokenizer = init_prompt_model(
    model_name=model_name,
    prompt_config=config["prompt_config"]
)

model = torch.nn.DataParallel(model)
graph_based = "graph" in config["context"]["context_type"]

data_dict = {
    "label": [],
    "neighbors": [],
    "alt-names": []
}

for node in gnd_graph.nodes(data=True):
    idn, data = node
    name = gnd_graph.pref_label_name(idn)
    neighbors = [gnd_graph.pref_label_name(node_idn) for node_idn in gnd_graph.neighbors(idn)]
    alt_names = gnd_graph.alt_label_names(idn)
    if neighbors or alt_names:
        data_dict["label"].append(name)
        data_dict["neighbors"].append(neighbors)
        data_dict["alt-names"].append(alt_names)

ds = Dataset.from_dict(data_dict)
ds = ds.train_test_split(test_size=0.05, seed=42, shuffle=True)
# Split the dataset into train, validation, and test sets
train_ds = ds["train"]
valid_ds = ds["test"]

data_collator = GraphDataCollator(
    tokenizer=tokenizer,
    config=config,
    device=DEVICE,
    neighbors=graph_based
)

if dev:
    # For development, use a smaller subset of the dataset
    train_ds = train_ds.select(range(10_000))

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
      project="xmlc-knowledge-general",
      name=f"{exp_name}",
      # Track hyperparameters and run metadata
      config={
          "model_name": model_name,
          "config": config,
          "n_train": len(train_ds),
          "params": num_trained_params,
      })

trainer = Trainer(config, data_collator)
trainer.train(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    output_dir=output_dir
)
