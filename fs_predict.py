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
from utils import process_output, map_labels, load_config
from prompt_str import SYSTEM_PROMPT, USER_PROMPT, FS_PROMPT

def few_shot_string(corpus, indices):
    """
    Generate a few-shot string based on the corpus and indices.
    """
    fs_examples = [
        FS_PROMPT.format(
            corpus[int(i)]["title"],
            "; ".join(corpus[int(i)]["label-names"]),
        ) for i in indices
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
parser.add_argument("--map_model", help="Sentence model for mapping", default="BAAI/bge-m3")
parser.add_argument(
    "--example-type", 
    type=str,
    help="How to choose few shot examples.", 
    default="title", 
    choices=("title", "label", "random"))
parser.add_argument("--dev", action='store_true', default=False)

arguments = parser.parse_args()
config_path = arguments.config
result_dir = arguments.result_dir
split = arguments.split
index_path = arguments.index
mapping_path = arguments.mapping
dev = arguments.dev
map_model = arguments.map_model
example_type = arguments.example_type

set_seed(arguments.seed)

print(f"Sampling examples using {example_type} strategy.")

# Load config 
config = load_config(config_path)

exp_name = config["experiment_name"]
model_name = config["model_name"]
best = config["context"]["best_example"]

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
    retriever_model=map_model,
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
train_ds = gnd_ds["train"]
test_ds = gnd_ds["test"]

if dev:
    test_ds = test_ds.select(range(10))

if example_type == "title": # Title means based on similarity of titles.
    # Generate mapping and index to retriever few short examples.
    print("Loading embedding model.")
    s_transf = SentenceTransformer(
        config["sentence_transformer_model"]
    )
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
elif example_type == "label":
    label_retriever = Retriever(
            retriever_model=config["sentence_transformer_model"],
            graph=gnd_graph,
            device=DEVICE)
    label_retriever.fit()
    label_doc_dict = {}
    for idx, instance in tqdm(enumerate(train_ds), desc="Create few-shot mapping..", total=train_ds.num_rows):
        doc_idn = instance["doc_idn"]
        for idn in instance["label-ids"]:
            if idn not in label_doc_dict:
                label_doc_dict[idn] = set()
            label_doc_dict[idn].add(idx)
    if best:
        print("Creating train title embeddings.")
        title_str = list(train_ds["title"])
        train_title_embed = retriever.retriever.encode(title_str, show_progress_bar=True, batch_size=1024)

pipe = pipeline(
        "text-generation",
        model=config["model_name"],
        torch_dtype=torch.bfloat16,
        device=DEVICE,
    )

print("Generating predictions.")
raw_predictions = []
n_examples = config["context"]["top_k"]
hops = config["context"]["hops"]

for row in tqdm(test_ds, total=test_ds.num_rows):
    title = row["title"]
    if example_type == "title":
        title_emb = s_transf.encode([title])
        fs_distance, fs_indices = index.search(title_emb, n_examples)
        fs_indices = fs_indices[0]
    elif example_type == "random":
        fs_indices = random.sample(range(train_ds.num_rows), k=n_examples)
    elif example_type == "label":
        label_idns = label_retriever.retrieve_with_neighbors(
        texts=[title],
        top_k=n_examples,
        k=hops
        )
        fs_indices = []
        for idn in label_idns[0]:
            idn_docs = list(label_doc_dict.get(idn, []))
            if idn_docs:
                if best:
                    input_embed = retriever.retriever.encode([title])
                    fs_docs_embed = train_title_embed[idn_docs]
                    sim = retriever.retriever.similarity(input_embed, fs_docs_embed).squeeze()
                    max_sim_index = torch.argmax(sim)
                    fs_ex = idn_docs[max_sim_index] 
                else:
                    fs_ex = random.choice(idn_docs)
                fs_indices.append(fs_ex)
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
processed_predictions = [process_output(pred) for pred in raw_predictions]

print("Mapping predictions to label space.")
# Map the labels to GND IDs
pred_idns = map_labels(
    processed_predictions, 
    retriever=retriever,
)

pred_df = pd.DataFrame(
    {
        "predictions": pred_idns,
        "raw_predictions": raw_predictions,
        "doc_idn": test_ds["doc_idn"],
        "title": test_ds["title"],
        "label-ids": test_ds["label-ids"]
    }
)
reranker = BGEReranker("BAAI/bge-reranker-v2-m3", device=DEVICE)
pred_df = reranker.rerank(
    pred_df,
    gnd_graph,
    bs=200
)
chp_str = f"few-shot"
pred_df.to_csv(os.path.join(result_dir, f"predictions-{split}-{chp_str}-seed-{arguments.seed}.csv"))

