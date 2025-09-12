import argparse
import os
import pickle
from random import sample

from datasets import Dataset
from sentence_transformers import SentenceTransformer,  losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from tqdm import tqdm
import yaml

from gnd_dataset import GNDDataset
from gnd_graph import GNDGraph

parser = argparse.ArgumentParser(description="Train a retriever model.")
parser.add_argument("--config", type=str, help="Path to the configuration file.")

arguments = parser.parse_args()
config_path = arguments.config

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

gnd_path = config["graph_path"]
model_name = config["sentence_transformer_model"]
num_examples = int(config["num_examples"])
num_epochs = int(config["num_epochs"])
lr = float(config["learning_rate"])
batch_size = int(config["batch_size"])
exp_name = config["experiment_name"]
out_dir = config["checkpoint_path"]

def build_fs_dataset(graph, dataset, n_examples=3):
    """
    Builds a dataset where a title is mapped to other title which share the same labels.

    Args:
        graph: networkx DiGraph 
        dataset: Huggingface dataset.
        n_examples: number of positive examples for each title.
    
    Returns:
        Huggingface dataset with positive and anchor keys.
    """
    # Create mapping from label idn to document idx
    # and mapping from doc idx to title string.
    label2docs= {}
    idx2doc = {}
    for node in graph.nodes():
        label2docs[node] = []
    for idx, instance in tqdm(enumerate(dataset)):
        labels = instance["label-ids"]
        for l in labels:
            if l not in label2docs:
                label2docs[l] = []
            label2docs[l].append(idx)
        idx2doc[idx] = instance["title"]
    # Create dataset for retriever training where an instance is mapped to its
    retriever_dict = {
    "anchor": [],
    "positive": [],
    }
    for label_id, docs in tqdm(label2docs.items()):
        for d in docs:
            title = idx2doc[d]
            n_sample = min(len(docs), n_examples)
            associated_examples = sample(docs, n_sample)
            for ex in associated_examples:
                ex_title = idx2doc[ex]
                retriever_dict["anchor"].append(title)
                retriever_dict["positive"].append(ex_title)
    return Dataset.from_dict(retriever_dict)

gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

gnd_ds = GNDDataset(
    data_dir=config["dataset_path"],
    gnd_graph=gnd_graph,
    load_from_disk=True,
)

train_ds = gnd_ds["train"]
model = SentenceTransformer(model_name)

eval_ds = gnd_ds["validate"].select(range(1000))

train_dataset = build_fs_dataset(graph=gnd_graph, dataset=train_ds, n_examples=num_examples).shuffle()
eval_dataset = build_fs_dataset(graph=gnd_graph, dataset=eval_ds, n_examples=num_examples)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=os.path.join(out_dir, exp_name),
    # Optional training parameters:
    num_train_epochs=2,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,
    eval_strategy="steps",
    eval_steps=1000,
    # Optional tracking/debugging parameters:
    save_total_limit=2,
    logging_steps=500,
    run_name=exp_name,  # Will be used in W&B if `wandb` is installed
)

loss = losses.MultipleNegativesRankingLoss(model)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    eval_dataset=eval_dataset,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()

