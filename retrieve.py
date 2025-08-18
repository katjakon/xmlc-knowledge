import argparse
import os
import pickle
import re
import yaml

import pandas as pd
import torch

from gnd_dataset import GNDDataset
from retriever import Retriever
from utils import get_label_mapping, get_title_mapping

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--result_dir", type=str, help="Path to the result directory.")
parser.add_argument("--index", type=str, help="Path to the index file.", default=None)
parser.add_argument("--label_mapping", type=str, help="Path to label mapping file", default=None)
parser.add_argument("--hops", type=int, default=None, help="Number of hops for neighbors.")
parser.add_argument("--relation", type=str, default=None, help="Relation for neighbors.")
parser.add_argument("--alt_labels", type=bool, default=False, help="Use alternative labels for mapping.")
parser.add_argument("--title_wise", type=bool, default=False, help="Use Title-to-Title mapping.")

arguments = parser.parse_args()
config_path = arguments.config
result_dir = arguments.result_dir
index_path = arguments.index
label_mapping_path = arguments.label_mapping
hops = arguments.hops
relation = arguments.relation
use_alt_labels = arguments.alt_labels
use_title_wise = arguments.title_wise

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

exp_name = config["experiment_name"]
batch_size = 1024

result_dir = os.path.join(result_dir, exp_name)

if os.path.exists(result_dir):
    print(f"Result directory {result_dir} already exists. Please remove it or choose a different name.")
    exit(1)
os.makedirs(result_dir)

# Load GND graph
gnd_path = config["graph_path"]
gnd_graph = pickle.load(open(gnd_path, "rb"))

data_dir = config["dataset_path"]
# Load GND dataset
gnd_ds = GNDDataset(
    data_dir=data_dir,
    gnd_graph=gnd_graph,
    config=config,
    load_from_disk=True,
)
test_ds = gnd_ds["test"]
train_ds = gnd_ds["train"]

retriever_model = config["sentence_transformer_model"]
retriever = Retriever(
    retriever_model=retriever_model,
    device=DEVICE,
)


if index_path is not None and label_mapping_path is not None:
    index = pickle.load(open(index_path, "rb"))
    mapping = pickle.load(open(label_mapping_path, "rb"))
else:
    if use_title_wise:
        strings, mapping = get_title_mapping(train_ds)
    else:
        strings, mapping = get_label_mapping(gnd_graph, use_alt_labels=use_alt_labels)
    index = retriever.fit(labels=strings, batch_size=batch_size)

if hops is not None:
    idns = retriever.retrieve_with_neighbors(
        graph=gnd_graph,
        mapping=mapping,
        index=index,
        texts=test_ds["title"],
        k=hops,
        top_k=10,
        batch_size=batch_size,
        relation=relation,
        title_wise=use_title_wise,
    )
else:
    sim, idns = retriever.retrieve(
        mapping=mapping,
        texts=test_ds["title"],
        top_k=10,
        batch_size=batch_size,
        index=index,
        title_wise=use_title_wise,
        )

pred_df = pd.DataFrame(
    {
        "predictions": idns,
        "label-ids": test_ds["label-ids"],
        "label-names": test_ds["label-names"],
        "title": test_ds["title"],
    }
)

pred_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)
