from ast import literal_eval
import pickle

import pandas as pd

from gnd_graph import GNDGraph
from gnd_dataset import GNDDataset

def add_entity_info(df, graph):
    df["entity"] = df["label-id"].apply(
    lambda x: graph.node_type(x) 
    )
    return df

def add_label_freq_info(df, freq_dict):
    df["label-freq"] = df["label-id"].apply(
        lambda x: freq_dict.get(x, 0)
        )

    df["label-freq"] = pd.cut(
        df["label-freq"], 
        bins=[-1, 0, 1, 10, 100, 1000, 100_000_000],
        labels=["x=0", "x=1", "1<x<=10", "10<x<=100", "100<x<=1000", "x>1000"]
        )
    return df

def add_hsg_info(df, mapping):
    df["hsg"] = df["doc_idn"].apply(lambda x: mapping.get(x))
    return df

def eval_by(pred_df, gold_df, by):
    grouped_by = pred_df.groupby(by)
    n_gold_by = gold_df.groupby(by)["label-id"].size()
    n_pred_by = grouped_by["correct"].size()
    correct_by = grouped_by["correct"].sum()
    prec = correct_by / n_pred_by
    rec = correct_by / n_gold_by
    f1 = 2*(prec*rec) / (prec+rec)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

file = "results/few-shot-baseline-8b/predictions-test-few-shot-seed-42.csv"

test_df = pd.read_csv(file)#.head(2)

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

for index, record in test_df.iterrows():
    gold_set = set(record["label-ids"])
    pred_set = set(record["reranked-predictions"])

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


long_df = pd.DataFrame.from_dict(long_dict)
gold_df = pd.DataFrame.from_dict(gold_dict)


for at_k in [1, 3, 5]:
    rank_df = long_df[long_df["rank"] <= at_k]
    print(f"@{at_k}")
    res = eval_by(rank_df, gold_df, by="doc_idn")
    res = {key: value.mean() for key, value in res.items()}
    print(res)

# Map and evaluate with entity types:
gnd_path = "gnd/gnd.pickle"
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)
long_df = add_entity_info(long_df, gnd_graph)
gold_df = add_entity_info(gold_df, gnd_graph)

by_entity = eval_by(long_df, gold_df, by="entity")
print(by_entity)


# Evaluate with label frequency
ds = GNDDataset(
    data_dir="dataset",
    gnd_graph=gnd_graph,
    load_from_disk=True
)
label_freq = ds.label_frequency(
    ds["train"]["label-ids"],
)

long_df = add_label_freq_info(long_df, label_freq)
gold_df = add_label_freq_info(gold_df, label_freq)

by_freq = eval_by(long_df, gold_df, by="label-freq")
print(by_freq)

path = "gnd/hsg-mapping-small.csv"
df = pd.read_csv(path)
df["hsg"] = df["hsg"].str[:1]

docid2hsg = {
    doc_id: hsg_code for doc_id, hsg_code in zip(df["doc_id"], df["hsg"])
}

long_df = add_hsg_info(long_df, docid2hsg)
gold_df = add_hsg_info(gold_df, docid2hsg)

res = eval_by(long_df, gold_df, by=["doc_idn", "hsg"])
res = {key: value.groupby("hsg").mean().to_dict() for key, value in res.items()}
print(res)
mapping = {
    "0": "General, Computer Science, Information Science",
    "1": "Philosophy and Psychology",
    "2": "Religion",
    "3": "Social Sciences",
    "4": "Language",
    "5": "Natural sciences and mathematics",
    "6": "Technology, medicine, applied sciences",
    "7": "Arts and entertainment",
    "8": "Literature",
    "9": "History and Geography",
    "B": "Fiction",
    "S": "Schoolbooks",
    "K": "Children's and Young Adult Literature"
}

