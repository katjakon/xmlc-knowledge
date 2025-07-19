from retriever import Retriever
from utils import get_label_mapping
import torch

import os
import pickle
import yaml

from gnd_dataset import GNDDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
gnd_path = "data/gnd.pickle"

# Load config 
with open("configs/config_retrieval.yml", "r") as f:
    config = yaml.safe_load(f)
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_ds = GNDDataset(
    data_dir="data/title",
    gnd_graph=gnd_graph,
    config=config
)
train_ds = gnd_ds["train"]


# Map raw labels to GND labels
#label_strings, label_mapping = get_label_mapping(gnd_graph)
strings, mapping = get_label_mapping(gnd_graph)

retriever_model = "retriever/testing/checkpoint-164204"
retriever = Retriever(
    retriever_model=retriever_model,
    device=DEVICE,
)

index = retriever.fit(labels=strings, batch_size=1024)

# Save the index to a file
index_path = os.path.join("search_indices", "ft_label_index.pkl")
with open(index_path, "wb") as f:
    pickle.dump(index, f)

# Save label_mapping to a file
mapping_path = os.path.join("search_indices", "ft_label_mapping.pkl")
with open(mapping_path, "wb") as f:
    pickle.dump(mapping, f)