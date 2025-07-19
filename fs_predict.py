import argparse
import os
import pickle
import re
import yaml

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import transformers
from transformers import  pipeline, set_seed
from tqdm import tqdm

from gnd_dataset import GNDDataset
from retriever import Retriever
from utils import load_model, generate_predictions, map_labels, process_output, SEP_TOKEN
from prompt_str import SYSTEM_PROMPT, USER_PROMPT, FS_PROMPT

def few_shot_string(corpus, indices):
    """
    Generate a few-shot string based on the corpus and indices.
    """
    fs_examples = [
        FS_PROMPT.format(
            corpus[int(i)]["title"],
            "; ".join(corpus[int(i)]["label-names"]),
        ) for i in indices[0]
    ]
    return fs_examples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--result_dir", type=str, help="Path to the result directory.")
parser.add_argument("--split", type=str, help="Split to use for evaluation.", default="test")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

arguments = parser.parse_args()
config_path = arguments.config
result_dir = arguments.result_dir
split = arguments.split

set_seed(arguments.seed)

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

exp_name = config["experiment_name"]
model_name = config["model_name"]

result_dir = os.path.join(result_dir, exp_name)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Load GND graph
gnd_path  = config["graph_path"]
gnd_graph = pickle.load(open(gnd_path, "rb"))

gnd_ds = GNDDataset(
    data_dir=config["dataset_path"],
    gnd_graph=gnd_graph,
    config=config,
    load_from_disk=True,
)
train_ds = gnd_ds["train"] 

# Generate mapping and index to retriever few short examples.
s_transf = SentenceTransformer(
    config["sentence_transformer_model"]
)
mapping = {}
strings = []

for i, instance in tqdm(enumerate(train_ds), total=train_ds.num_rows):
    title = instance["title"]
    strings.append(title)
    mapping[i] = title

embeddings = s_transf.encode(
    strings,
    show_progress_bar=True,
    batch_size=1024,
)

index = faiss.IndexHNSWFlat(
    s_transf.get_sentence_embedding_dimension(),
    200,  # M parameter for HNSW
)
index.add(embeddings)

pipe = pipeline(
        "text-generation",
        model=config["model_name"],
        torch_dtype=torch.bfloat16,
        device=DEVICE,
    )
test_ds = gnd_ds[split]
raw_predictions = []

for row in tqdm(test_ds, total=test_ds.num_rows):
    title = row["title"]
    title_emb = s_transf.encode([title])
    fs_distance, fs_indices = index.search(title_emb, 3)
    fs_ex = few_shot_string(train_ds, fs_indices)	
    system_prompt = f"{SYSTEM_PROMPT} {'\n'.join(fs_ex)}"
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": USER_PROMPT.format(row["title"])},
    ]
    outputs = pipe(messages, num_return_sequences=1, do_sample=True, temperature=0.7)
    new_tokens = outputs[0]["generated_text"][-1]["content"]
    raw_predictions.append(new_tokens.strip())

pred_df = pd.DataFrame(
    {
        "raw_predictions": raw_predictions,
        "label-ids": test_ds["label-ids"],
        "label-names": test_ds["label-names"],
        "title": test_ds["title"],
    }
)

chp_str = f"few-shot"
pred_df.to_csv(os.path.join(result_dir, f"predictions-{split}-{chp_str}-seed-{arguments.seed}.csv"))