import argparse
import os
import pickle
import yaml

import torch

from retriever import Retriever
from gnd_dataset import GNDDataset
from gnd_graph import  GNDGraph

from utils import get_title_mapping, load_config
parser = argparse.ArgumentParser(
    description="Fit a vector search index.")
parser.add_argument("--config", help="Path to yaml config file where embedding model is specified.")
parser.add_argument("--name", help="Name of search indices")
parser.add_argument("--out_dir", help="Path to output dir, will be created if it does not exist.", default="search_indices")

args = parser.parse_args()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config_path = args.config
name = args.name
out_dir = args.out_dir

if not os.path.exists(out_dir):
    os.path.mkdir(out_dir)

# Load config 
config = load_config(config_path)
title_wise = config["context"]["title_wise"]
gnd_graph = pickle.load(open(config["graph_path"], "rb"))
gnd_graph = GNDGraph(gnd_graph)
gnd_ds = GNDDataset(
    data_dir="dataset",
    gnd_graph=gnd_graph,
    config=config, 
    load_from_disk=True
)
train_ds = gnd_ds["train"]
dev = False

if dev:
    train_ds = train_ds.select(range(10))

retriever_model = config["sentence_transformer_model"]
retriever = Retriever(
    retriever_model=retriever_model,
    graph=gnd_graph,
    device=DEVICE,
)

if title_wise:
    strings, mapping = get_title_mapping(train_ds)
    index = retriever.fit_title_wise(title_strings=strings, title_mapping=mapping)
else:
    index = retriever.fit(batch_size=1024)

# Save the index to a file
index_path = os.path.join("search_indices", f"{name}-index.pkl")
mapping = retriever.mapping

with open(index_path, "wb") as f:
    pickle.dump(index, f)

# Save label_mapping to a file
mapping_path = os.path.join("search_indices", f"{name}-mapping.pkl")
with open(mapping_path, "wb") as f:
    pickle.dump(mapping, f)
