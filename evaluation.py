import argparse
from ast import literal_eval
import pickle
import yaml
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import logging
from sentence_transformers import SentenceTransformer

from evaluate_results.similarity import get_similarity_data, bin_similarity
from evaluate_results.entity_types import add_entity_info
from evaluate_results.mt_metrics import gold_label_strings, bertscore_results, meteor_results, rouge_results
from evaluate_results.label_frequency import add_label_freq_info
from evaluate_results.genres import add_hsg_info, hsg_data
from reranker import BGEReranker
from retriever import Retriever
from gnd_dataset import GNDDataset
from gnd_graph import GNDGraph
from utils import inverse_distance_weight, map_labels, process_output

logging.set_verbosity_error()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def best_distance_weight(pred_label, gold_labels, graph):
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

parser = argparse.ArgumentParser(description="Evaluate the model on the GND dataset.")
parser.add_argument("--reranker_model", type=str, help="Name of the reranker model.", default="BAAI/bge-reranker-v2-m3")
parser.add_argument("--predictions_file", type=str, help="Path to the predictions file.")
parser.add_argument("--write_reranked", type=bool, default=True, help="Whether to write the reranked predictions to the given file.")
parser.add_argument("--dataset_path", help="Path to dataset", default="dataset")
parser.add_argument("--gnd_path", help="Path to gnd file", default="gnd/gnd.pickle")
parser.add_argument("--sentence_model", help="String that defines sentence transformer", default="BAAI/bge-m3")
parser.add_argument("--force_remap", help="Map raw predictions again.", action="store_true", default=False)
parser.add_argument("--index", help="Path to the index file.", default=None)
parser.add_argument("--mapping", help="Path to the mapping file.", default=None)

arguments = parser.parse_args()
reranker_str = arguments.reranker_model
pred_file = arguments.predictions_file
write_reranked = arguments.write_reranked
ds_path = arguments.dataset_path
gnd_path = arguments.gnd_path
sentence_transformer_str = arguments.sentence_model
force_remap = arguments.force_remap
index = arguments.index
mapping = arguments.mapping

reranker = BGEReranker(reranker_str, device=DEVICE)
sentence_model = SentenceTransformer(sentence_transformer_str)

# Load GND graph
gnd_path = gnd_path
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

# Load predictions file.
test_df = pd.read_csv(pred_file)

ds = GNDDataset(
    data_dir=ds_path,
    gnd_graph=gnd_graph,
    load_from_disk=True
)

full_eval_metrics = {}
# Convert to list
test_df["predictions"] = test_df["predictions"].apply(literal_eval)
test_df["label-ids"] = test_df["label-ids"].apply(literal_eval)

if force_remap:
    print("Re-doing mapping step.")
    retriever = Retriever(
        retriever_model=sentence_transformer_str,
        graph=gnd_graph,
        device=DEVICE,
    )
    if index is None or mapping is None:
        raise ValueError("Need mapping and index!")
    retriever.load_search_index(
        index_path=index,
        mapping_path=mapping,
    )
    raw_predictions = test_df["raw_predictions"]
    processed_predictions = []
    for pred_str in raw_predictions:
        pred_str = process_output(pred_str)
        processed_predictions.append(pred_str)

    mapped_predictions = map_labels(
        prediction_list=processed_predictions,
        retriever=retriever
    )
    reranker = BGEReranker("BAAI/bge-reranker-v2-m3", device=DEVICE)
    test_df = reranker.rerank(
        test_df,
        gnd_graph,
        bs=200
        )
    test_df.to_csv(pred_file, index=False)

if "reranked-predictions" not in test_df.columns or "scores" not in test_df.columns:
    reranker = BGEReranker(reranker_str, device=DEVICE)
    test_df = reranker.rerank(
        test_df,
        gnd_graph,
        bs=200
    )
    # Save reranked_df
    if write_reranked:
        test_df.to_csv(pred_file, index=False)
else:
    test_df["reranked-predictions"] = test_df["reranked-predictions"].apply(literal_eval)

long_dict = {
    "doc_idn": [],
    "label-id": [],
    "score": [],
    "rank": [],
    "correct": [],
    "inverse-distance": [],
    "similarity": []
}
gold_dict = {
    "doc_idn": [],
    "label-id": [],
    "similarity": []
}

sim_data = get_similarity_data(
    sentence_model=sentence_model,
    data=test_df, 
    gnd_graph=gnd_graph, 
    batch_size=512
)

for index, record in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    gold_set = set(record["label-ids"])
    pred_set = set(record["reranked-predictions"])

    title_embedding = sim_data["title_embeddings"][index]
    for pred_idx, (pred, score) in enumerate(zip(record["reranked-predictions"], record["scores"])):
        rank = pred_idx + 1
        long_dict["doc_idn"].append(record["doc_idn"])
        long_dict["label-id"].append(pred)
        long_dict["score"].append(score)
        long_dict["rank"].append(rank)
        long_dict["correct"].append(pred in gold_set)

        # Compute the score for the shortest distance of prediction and gold label
        distance_score = best_distance_weight(pred, record["label-ids"], gnd_graph)
        long_dict["inverse-distance"].append(distance_score)

        # How similar is the predicted label to the title?
        label_idx = sim_data["idn2idx"].get(pred)
        sim = 0
        if label_idx is not None:
            pred_label_embedding = sim_data["label_embeddings"][label_idx]
            sim = sentence_model.similarity(title_embedding, pred_label_embedding).item()
        long_dict["similarity"].append(sim)

    for gold in record["label-ids"]:
        gold_dict["doc_idn"].append(record["doc_idn"])
        gold_dict["label-id"].append(gold)
        # How similar is the gold label to the title?
        label_idx = sim_data["idn2idx"].get(gold)
        sim = 0
        if label_idx is not None:
            gold_label_embedding = sim_data["label_embeddings"][label_idx]
            sim = sentence_model.similarity(title_embedding, gold_label_embedding).item()
        gold_dict["similarity"].append(sim)


long_df = pd.DataFrame.from_dict(long_dict)
gold_df = pd.DataFrame.from_dict(gold_dict)

print("Inverse Distance Metric")
distance_metric = long_df.groupby("doc_idn")["inverse-distance"].mean()
print(distance_metric.mean())
full_eval_metrics["weighted_precision"] = float(distance_metric.mean())

## PERFORMANCE FOR ALL PREDICTIONS
result_no_limit = eval_by(long_df, gold_df, by="doc_idn").mean()
full_eval_metrics.update(result_no_limit.to_dict())
print("Performance for all predictions: ")
print(result_no_limit)

# PERFORMANCE AT K
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
print("Perfomance by label frequency: ")
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

## Similarity
long_df = bin_similarity(long_df)
gold_df = bin_similarity(gold_df)

by_sim = eval_by(long_df, gold_df, by="similarity")
full_eval_metrics["similarity"] = by_sim.to_dict("index")
print(by_sim)

## GENRES
print("Perfomance by document genres: ")
path = "gnd/hsg-mapping-small.csv"
docid2hsg, hsg2label = hsg_data(path, shorten_codes=True)

long_df = add_hsg_info(long_df,  docid2hsg=docid2hsg, hsg2label=hsg2label)
gold_df = add_hsg_info(gold_df, docid2hsg=docid2hsg, hsg2label=hsg2label)

by_genres = eval_by(long_df, gold_df, by="hsg", include_counts=False)
full_eval_metrics["genres"] = by_genres.to_dict("index")
print(by_genres)

# MT Metrics
gold_labels = gold_label_strings(test_df, gnd_graph)

if "raw_predictions" in test_df.columns:
    raw_preds = test_df["raw_predictions"]

    bert_results = bertscore_results(pred_strings=raw_preds, gold_strings=gold_labels)
    print(f"BERTScore: {bert_results}")
    full_eval_metrics["bertscore"] = bert_results

    meteor = meteor_results(pred_strings=raw_preds, gold_strings=gold_labels)
    print(f"Meteor: {meteor}")
    full_eval_metrics["meteor"] = meteor

    rouge = rouge_results(pred_strings=raw_preds, gold_strings=gold_labels)
    print(f"Rouge: {rouge}")
    full_eval_metrics["rouge"] = rouge


# Save all metrics to a YAML file
file_name = os.path.basename(pred_file).split(".")[0]
eval_path = os.path.join(os.path.dirname(pred_file), f"{file_name}_eval.yaml")
with open(eval_path, "w") as f:
    yaml.dump(full_eval_metrics, f, indent=2, sort_keys=False, allow_unicode=True)

