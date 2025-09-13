from ast import literal_eval
import pickle

import pandas as pd

from gnd_graph import GNDGraph
from gnd_dataset import GNDDataset

def precision_at_k(y_true, y_pred, k=None):
    if k is not None:
        y_pred = y_pred[:k]
    correct = len(set(y_true).intersection(set(y_pred)))
    return correct / len(y_pred) if len(y_pred) > 0 else 0

def recall_at_k(y_true, y_pred, k=None):
    if k is not None:
        y_pred = y_pred[:k]
    correct = len(set(y_true).intersection(set(y_pred)))
    return correct / len(y_true)

file = "results/few-shot-baseline-8b/predictions-test-few-shot-seed-42.csv"

test_df = pd.read_csv(file).head(2)

test_df["predictions"] = test_df["predictions"].apply(literal_eval)
test_df["label-ids"] = test_df["label-ids"].apply(literal_eval)
test_df["reranked-predictions"] = test_df["reranked-predictions"].apply(literal_eval)
test_df["scores"] = test_df["scores"].apply(literal_eval)

long_dict = {
    "doc_idn": [],
    "label-id": [],
    "score": [],
    "rank": [],
    "correct": [],
}
gold_dict = {
    "doc_idn": [],
    "label-id": []
}
at = 1
recall = []
for index, record in test_df.iterrows():
    gold_set = set(record["label-ids"])
    pred_set = set(record["reranked-predictions"])
    recall.append(recall_at_k(y_pred=record["reranked-predictions"], y_true=record["label-ids"], k=at))
    
    for pred_idx, (pred, score) in enumerate(zip(record["reranked-predictions"], record["scores"])):
        rank = pred_idx + 1
        long_dict["doc_idn"].append(record["doc_idn"])
        long_dict["label-id"].append(pred)
        long_dict["score"].append(score)
        long_dict["rank"].append(rank)
        long_dict["correct"].append(pred in gold_set)
    
    for gold in record["label-ids"]:
        gold_dict["doc_idn"].append(record["doc_idn"])
        gold_dict["label-id"].append(gold)

at = 1
long_df = pd.DataFrame.from_dict(long_dict)
gold_df = pd.DataFrame.from_dict(gold_dict)

rank_df = long_df[long_df["rank"] <= at ]
correct_by_doc = rank_df.groupby("doc_idn")["correct"].mean()

n_gold_by_doc = gold_df.groupby("doc_idn")["label-id"].nunique()
precision = correct_by_doc.mean()
recall = (correct_by_doc/n_gold_by_doc).mean()
print("Precision", correct_by_doc.mean())
print("Recall", (correct_by_doc/n_gold_by_doc).mean())

# Map with entity types:
gnd_path = "gnd/gnd.pickle"
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

# Add entity information.
long_df["entity"] = long_df["label-id"].apply(
    lambda x: gnd_graph.node_type(x) 
)
gold_df["entity"] = gold_df["label-id"].apply(
    lambda x: gnd_graph.node_type(x) 
)

grouped_entity = long_df.groupby("entity")
n_gold_by_entity = gold_df.groupby("entity")["label-id"].size()
n_pred_by_entity = grouped_entity["correct"].size()
correct_by_entity = grouped_entity["correct"].sum()

print(correct_by_entity/n_pred_by_entity)
print(correct_by_entity/n_gold_by_entity)

# LABEL
ds = GNDDataset(
    data_dir="dataset",
    gnd_graph=gnd_graph,
    load_from_disk=True
)
label_freq = ds.label_frequency(
    ds["train"]["label-ids"],
)
