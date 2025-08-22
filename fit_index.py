import argparse
import os
import pickle
import yaml

import torch

from retriever import Retriever
from gnd_dataset import GNDDataset
from gnd_graph import  GNDGraph

parser = argparse.ArgumentParser(
    description="Fit a vector search index.")

parser.add_argument("--gnd_path", help="Path to GND pickle file.")
parser.add_argument("--config", help="Path to yaml config file where embedding model is specified.")
parser.add_argument("--name", help="Name of search indices")
parser.add_argument("--out_dir", help="Path to output dir, will be created if it does not exist.", default="search_indices")

args = parser.parse_args()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
gnd_path = args.gnd_path
config_path = args.config
name = args.name
out_dir = args.out_dir


if not os.path.exists(out_dir):
    os.path.mkdir(out_dir)

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)
gnd_ds = GNDDataset(
    data_dir="dataset",
    gnd_graph=gnd_graph,
    config=config, 
    load_from_disk=True
)
train_ds = gnd_ds["train"]

retriever_model = config["sentence_transformer_model"]
retriever = Retriever(
    retriever_model=retriever_model,
    graph=gnd_graph,
    device=DEVICE,
)

index = retriever.fit(batch_size=1024)

# Save the index to a file
index_path = os.path.join("search_indices", f"{name}-label_index.pkl")
mapping = retriever.mapping

with open(index_path, "wb") as f:
    pickle.dump(index, f)

# Save label_mapping to a file
mapping_path = os.path.join("search_indices", f"{name}-label_mapping.pkl")
with open(mapping_path, "wb") as f:
    pickle.dump(mapping, f)
