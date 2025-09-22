from ast import literal_eval
import pickle

import pandas as pd
from tqdm import tqdm

from gnd_graph import GNDGraph
from gnd_dataset import GNDDataset
from utils import inverse_distance_weight

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

def minimum_distance(pred_label, gold_labels, graph):
    max_weight = - float('inf')
    for g in gold_labels:
        weight = inverse_distance_weight(graph, g, pred_label)
        if weight > max_weight:
            max_weight = weight
    return max_weight

def eval_by(pred_df, gold_df, by, include_counts=True):
    grouped_by = pred_df.groupby(by)
    n_gold_by = gold_df.groupby(by)["label-id"].size()
    n_pred_by = grouped_by["label-id"].size()
    correct_by = grouped_by["correct"].sum()
    prec = correct_by / n_pred_by
    rec = correct_by / n_gold_by
    f1 = 2*(prec*rec) / (prec+rec)
    eval_df =  pd.DataFrame(
        data={"recall": rec.values,
        "precision": prec.values,
        "f1": f1.values}, 
        index=n_gold_by.index
    )
    if include_counts:
        eval_df["n_gold"] = n_gold_by.values
        eval_df["n_pred"] = n_pred_by.values
    return eval_df.fillna(0)

file = "results/few-shot-baseline-8b/predictions-test-few-shot-seed-42.csv"

test_df = pd.read_csv(file).head(20)

test_df["predictions"] = test_df["predictions"].apply(literal_eval)
test_df["label-ids"] = test_df["label-ids"].apply(literal_eval)
test_df["reranked-predictions"] = test_df["reranked-predictions"].apply(literal_eval)
test_df["scores"] = test_df["scores"].apply(literal_eval)

full_eval_metrics = {

}

# Load gnd graph:
gnd_path = "gnd/gnd.pickle"
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

long_dict = {
    "doc_idn": [],
    "label-id": [],
    "score": [],
    "rank": [],
    "correct": [],
    "inv-distance": []
}
gold_dict = {
    "doc_idn": [],
    "label-id": []
}

for index, record in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    gold_set = set(record["label-ids"])
    pred_set = set(record["reranked-predictions"])

    for pred_idx, (pred, score) in enumerate(zip(record["reranked-predictions"], record["scores"])):
        rank = pred_idx + 1
        long_dict["doc_idn"].append(record["doc_idn"])
        long_dict["label-id"].append(pred)
        long_dict["score"].append(score)
        long_dict["rank"].append(rank)
        long_dict["correct"].append(pred in gold_set)
        long_dict["inv-distance"].append(minimum_distance(pred, record["label-ids"], gnd_graph))
    
    for gold in record["label-ids"]:
        gold_dict["doc_idn"].append(record["doc_idn"])
        gold_dict["label-id"].append(gold)


long_df = pd.DataFrame.from_dict(long_dict)
gold_df = pd.DataFrame.from_dict(gold_dict)

print("Inverse Distance Metric")
distance_metric = long_df.groupby("doc_idn")["inv-distance"].mean()
print(distance_metric.mean())
full_eval_metrics["weighted_precision"] = float(distance_metric.mean())

# AT K.
for at_k in [1, 3, 5]:
    rank_df = long_df[long_df["rank"] <= at_k]
    print(f"Performance@{at_k}")
    res = eval_by(rank_df, gold_df, by="doc_idn", include_counts=False).mean()
    res.index = res.index + f"@{at_k}"
    print(res)
    full_eval_metrics.update(res.to_dict())

## ENTITY
print("Perfomance by entity type: ")
long_df = add_entity_info(long_df, gnd_graph)
gold_df = add_entity_info(gold_df, gnd_graph)

by_entity = eval_by(long_df, gold_df, by="entity")
full_eval_metrics["entity_types"] = by_entity.to_dict("index")
print(by_entity)

## LABEL FREQUENCY
print("Perfomance by label frquency: ")
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
full_eval_metrics["label_frequencies"] = by_freq.to_dict("index")
print(by_freq)

## GENRES
print("Perfomance by document genres: ")
path = "gnd/hsg-mapping-small.csv"
df = pd.read_csv(path)
df["hsg"] = df["hsg"].str[:1]

docid2hsg = {
    doc_id: hsg_code for doc_id, hsg_code in zip(df["doc_id"], df["hsg"])
}

long_df = add_hsg_info(long_df, docid2hsg)
gold_df = add_hsg_info(gold_df, docid2hsg)

by_genres = eval_by(long_df, gold_df, by="hsg", include_counts=False)
full_eval_metrics["genres"] = by_genres.to_dict("index")
print(by_genres)
print(full_eval_metrics)


