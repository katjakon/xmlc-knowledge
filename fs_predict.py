import argparse
import os
import pickle
import re
import yaml
import random

from datasets import Dataset
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import transformers
from transformers import  pipeline, set_seed
from tqdm import tqdm

from gnd_graph import GNDGraph
from gnd_dataset import GNDDataset
from reranker import BGEReranker
from retriever import Retriever
from utils import process_output, map_labels
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
parser.add_argument("--index", type=str, help="Path to the index file.")
parser.add_argument("--mapping", type=str, help="Path to the mapping file.")
parser.add_argument("--n_examples", type=int, help="Number of examples.", default=3)
parser.add_argument(
    "--example-type", 
    type=str,
    help="How to choose few shot examples.", 
    default="title", 
    choices=("title", "label", "random"))

arguments = parser.parse_args()
config_path = arguments.config
result_dir = arguments.result_dir
split = arguments.split
index_path = arguments.index
mapping_path = arguments.mapping
example_type = arguments.example_type
n_examples = arguments.n_examples

set_seed(arguments.seed)

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

exp_name = config["experiment_name"]
model_name = config["model_name"]

result_dir = os.path.join(result_dir, exp_name)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

print("Loading Graph.")
# Load GND graph
gnd_path  = config["graph_path"]
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

print("Loading retriever.")
retriever = Retriever(
    retriever_model=config["sentence_transformer_model"],
    graph=gnd_graph,
    device=DEVICE,
)

retriever.load_search_index(
    index_path=index_path,
    mapping_path=mapping_path,
)

print("Loading dataset.")
gnd_ds = GNDDataset(
    data_dir=config["dataset_path"],
    gnd_graph=gnd_graph,
    config=config,
    load_from_disk=True,
)
train_ds = gnd_ds["train"] #.select(range(100))
test_ds = gnd_ds["test"] #.select(range(10))

print("Loading embedding model.")
# Generate mapping and index to retriever few short examples.
s_transf = SentenceTransformer(
    config["sentence_transformer_model"]
)

if example_type == "title": # Title means based on similarity of titles.
    mapping = {}
    strings = []
    print("Embedding examples.")
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

print("Generating predictions.")
raw_predictions = []
for row in tqdm(test_ds, total=test_ds.num_rows):
    title = row["title"]
    if example_type == "title":
        title_emb = s_transf.encode([title])
        fs_distance, fs_indices = index.search(title_emb, n_examples)
    elif example_type == "random":
        fs_indices = random.sample(range(train_ds.num_rows), k=n_examples)
    else:
        raise ValueError(f"{example_type} is not implemented.")
    fs_ex = few_shot_string(train_ds, fs_indices)
    system_prompt = f"{SYSTEM_PROMPT} " + '\n'.join(fs_ex)
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": USER_PROMPT.format(row["title"])},
    ]
    outputs = pipe(messages, num_return_sequences=1, do_sample=True, temperature=0.7)
    new_tokens = outputs[0]["generated_text"][-1]["content"]
    raw_predictions.append(new_tokens.strip())

# Process the raw predictions to match the expected format
raw_predictions = [process_output(pred) for pred in raw_predictions]

print("Mapping predictions to label space.")
# Map the labels to GND IDs
pred_idns = map_labels(
    raw_predictions, 
    retriever=retriever,
)

pred_df = pd.DataFrame(
    {
        "predictions": pred_idns,
        "raw_predictions": raw_predictions,
        "doc_idn": test_ds["doc_idn"],
        "title": test_ds["title"],
    }
)

reranker = BGEReranker("BAAI/bge-reranker-v2-m3", device=DEVICE)
pred_df = reranker.rerank(
    pred_df,
    gnd_graph,
    bs=200
)

# # Reshape predictions necessary for QM.
# index_file = os.path.join("title-qm", "sci-ger-ideal.arrow")
# index_qm = pd.read_feather(index_file)


# doc_ids = {i: index_qm[index_qm["location"] == i]["idn"].values[0] for i in tqdm(range(pred_df.shape[0]), desc="Map indices to doc idn")}

# suggestions = []

# for i, row in tqdm(pred_df.iterrows(), total=pred_df.shape[0], desc="Reshaping predictions..."):
#     doc_id = doc_ids[i]
#     reranked_predictions = row["reranked-predictions"]
#     scores = row["scores"]
#     for rank, (label_id, score) in enumerate(zip(reranked_predictions, scores)):
#         suggestions.append({"doc_id": doc_id, "label_id": label_id, "score": score, "rank": rank + 1})
# suggestions_df = pd.DataFrame(suggestions)  # Convert to dataframe

chp_str = f"few-shot"
pred_df.to_csv(os.path.join(result_dir, f"predictions-{split}-{chp_str}-seed-{arguments.seed}.csv"))
# suggestions_df.to_feather(os.path.join(result_dir, f"suggestions.arrow"))
