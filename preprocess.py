import argparse
import os
import pickle

import torch
import yaml

from gnd_dataset import GNDDataset
from retriever import Retriever
from utils import init_tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Create a dataset.")
parser.add_argument("--data_dir", type=str, help="Path to the GND dataset directory.")
parser.add_argument("--config", type=str, help="Path to the configuration file.")

arguments = parser.parse_args()
data_dir = arguments.data_dir
config_path = arguments.config

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

exp_name = config["experiment_name"]
dataset_path = config["dataset_path"]
context_config = config["context"]
context = context_config["context_type"]
hops = context_config["hops"]
relation = context_config["relation"]
top_k = context_config["top_k"]
mapping_path = context_config["mapping_path"]
index_path = context_config["index_path"]
title_wise = context_config["title_wise"]


gnd_path  = config["graph_path"]
gnd_graph = pickle.load(open(gnd_path, "rb"))
tokenizer = init_tokenizer(config["model_name"])

retriever_model = config["sentence_transformer_model"]
retriever = Retriever(
    retriever_model=retriever_model,
    device=DEVICE,
)

gnd_ds = GNDDataset(
    data_dir=data_dir,
    gnd_graph=gnd_graph,
    config=config
)

if context is not None:
    mapping = pickle.load(open(mapping_path, "rb")) 
    index = pickle.load(open(index_path, "rb"))
    gnd_ds.add_context(
        retriever=retriever,
        tokenizer=tokenizer,
        index=index,
        mapping=mapping,
        hops=hops,
        relation=relation,
        k=top_k,
        context_type=context,
        use_title_wise=title_wise,
        batch_size=2048,
        )

# Tokenize the datasets
# gnd_ds.tokenize_datasets(tokenizer=tokenizer, splits=["train", "validate"])
# gnd_ds.inference_tokenize_datasets(tokenizer=tokenizer, splits=["test"])

# Save the datasets
gnd_ds.save_to_disk(dataset_path)