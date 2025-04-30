import argparse
import os
import pickle
import re
import yaml

import pandas as pd
import torch

from gnd_dataset import GNDDataset
from retriever import Retriever
from utils import get_label_mapping

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--data_dir", type=str, help="Path to the GND dataset directory.")
parser.add_argument("--gnd_graph", type=str, help="Path to the GND graph file (pickle).")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--result_dir", type=str, help="Path to the result directory.")
parser.add_argument("--name", type=str, help="Name of the experiment.")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for predictions.")
parser.add_argument("--index", type=str, help="Path to the index file.", default=None)
parser.add_argument("--label_mapping", type=str, help="Path to label mapping file", default=None)
parser.add_argument("--hops", type=int, default=None, help="Number of hops for neighbors.")
parser.add_argument("--relation", type=str, default=None, help="Relation for neighbors.")

arguments = parser.parse_args()
data_dir = arguments.data_dir
gnd_graph = arguments.gnd_graph
config_path = arguments.config
result_dir = arguments.result_dir
exp_name = arguments.name
batch_size = arguments.batch_size
index_path = arguments.index
label_mapping_path = arguments.label_mapping
hops = arguments.hops
relation = arguments.relation

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

result_dir = os.path.join(result_dir, exp_name)

if os.path.exists(result_dir):
    print(f"Result directory {result_dir} already exists. Please remove it or choose a different name.")
    exit(1)
os.makedirs(result_dir)


retriever_model = config["sentence_transformer_model"]
retriever = Retriever(
    retriever_model=retriever_model,
    device=DEVICE,
)

# Load GND graph
gnd_graph = pickle.load(open(gnd_graph, "rb"))
if index_path is not None and label_mapping_path is not None:
    index = pickle.load(open(index_path, "rb"))
    label_mapping = pickle.load(open(label_mapping_path, "rb"))
else:
    label_strings, label_mapping = get_label_mapping(gnd_graph)
    index = retriever.fit(labels=label_strings, batch_size=batch_size)

# Load GND dataset
gnd_ds = GNDDataset(
    data_dir=data_dir,
    gnd_graph=gnd_graph,
    config=config
)

test_ds = gnd_ds["test"]

if hops is not None:
    idns = retriever.retrieve_with_neighbors(
        graph=gnd_graph,
        mapping=label_mapping,
        index=index,
        texts=test_ds["title"],
        k=hops,
        top_k=10,
        batch_size=batch_size,
        relation=relation
    )
else:
    sim, idns = retriever.retrieve(
        mapping=label_mapping,
        texts=test_ds["title"],
        top_k=10,
        batch_size=batch_size,
        index=index)

pred_df = pd.DataFrame(
    {
        "predictions": idns,
        "label-ids": test_ds["label-idns"],
        "label-names": test_ds["label-names"],
        "title": test_ds["title"],
    }
)

pred_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)