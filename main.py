import argparse
import os
import pickle
import yaml

import torch
import pandas as pd
import wandb

from default_config import default_config
from gnd_dataset import GNDDataset
from gnd_graph import GNDGraph
from data_collator import DataCollator
from trainer import Trainer
from retriever import Retriever
from utils import init_prompt_model, load_model, generate_graph_data, get_label_embeddings, load_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--dev", action="store_true", help="Run in development mode with a smaller dataset.")
parser.add_argument("--load_from_pretrained", help="Path to a pretrained model to load from.", type=str, default=False)
parser.add_argument("--num_validate", default=2000)

arguments = parser.parse_args()
config_path = arguments.config
dev = arguments.dev
load_from_pretrained = arguments.load_from_pretrained
num_validate = arguments.num_validate
# Load config 
config = load_config(config_path)

label_mapping_path = config["label_mapping_path"]
exp_name = config["experiment_name"]
output_dir = config["checkpoint_path"]
output_dir = os.path.join(output_dir, exp_name)
prompt_config = config["prompt_config"]

if os.path.exists(output_dir):
    print(f"Output directory {output_dir} already exists. Please remove it or choose a different name.")
    exit(1)

os.makedirs(output_dir)

# Load GND graph
gnd_path  = config["graph_path"]
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

model_name = config["model_name"]

graph_based = False
if config["context"]["context_type"] is not None:
    graph_based = "graph" in config["context"]["context_type"]

if load_from_pretrained:
    # Load the model from a pretrained checkpoint
    keys_to_load = config.get("load_pt_keys", None)
    model, tokenizer = load_model(
        checkpoint_path=load_from_pretrained,
        config=config,
        device=DEVICE,
        load=keys_to_load,
    )
else:
    label_df = pd.read_feather(label_mapping_path)
    label_embeddings = None
    if graph_based:
        label_embeddings = get_label_embeddings(
            mapping_df=label_df, 
            prompt_config=prompt_config, 
            kind=prompt_config["kge_kind"], 
            sentence_transformer_model=prompt_config["kge_encoder"], 
            path=prompt_config["kge_path"], 
            device=DEVICE
        )
    model, tokenizer = init_prompt_model(
        model_name=model_name,
        prompt_config=prompt_config,
        tune_lm_head=True,
        embeddings=label_embeddings
    )
    model = torch.nn.DataParallel(model)

# Load GND dataset
data_dir = config["dataset_path"]
gnd_ds = GNDDataset(
    data_dir=data_dir,
    gnd_graph=gnd_graph,
    config=config, 
    load_from_disk=True,
)

# Split the dataset into train, validation, and test sets
train_ds = gnd_ds["train"]
valid_ds = gnd_ds["validate"]
if num_validate < valid_ds.num_rows:
    valid_ds = valid_ds.select(range(num_validate))
test_ds = gnd_ds["test"]

print("Number of validate examples: ", valid_ds.num_rows)

retriever_model = config["sentence_transformer_model"]
retriever = Retriever(
    retriever_model=retriever_model,
    graph=gnd_graph,
    device=DEVICE,
)

index_path = config["context"].get("index_path")
mapping_path = config["context"].get("mapping_path")

if (
    index_path is not None 
    and mapping_path is not None 
    and os.path.exists(index_path) 
    and os.path.exists(mapping_path)
    ):
    retriever.load_search_index(
        mapping_path=mapping_path,
        index_path=index_path
    )
else:
    retriever.fit(batch_size=1000)

data_collator = DataCollator(
        tokenizer=tokenizer,
        graph=gnd_graph,  
        device=DEVICE,
        use_context=config["context"]["context_type"] is not None,
        top_k=config["context"]["top_k"],
        hops=config["context"]["hops"],
        retriever=retriever, 
        graph_based=graph_based
    )
if graph_based: 
    idn2idx, idx2idn, pyg_data = generate_graph_data(
        label_mapping_path=label_mapping_path,
        graph=gnd_graph
    )
    data_collator.add_graph_data(idn2idx=idn2idx, idx2idn=idx2idn, pyg_data=pyg_data)
    
if dev:
    # For development, use a smaller subset of the dataset
    train_ds = train_ds.select(range(10_000))

# How many parameters are in the model?
total_model_params = 0
num_trained_params = 0

for n, p in model.named_parameters():
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
prompt_type = prompt_config["type"]
layer = prompt_config["at_layer"]
num_tokens = prompt_config["num_prompt_tokens"]

project_name = "xmlc-knowledge-final"
if dev:
    project_name = project_name + "-dev"
wandb.init(
      # Set the project where this run will be logged
      project=project_name,
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
