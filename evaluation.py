import argparse
from ast import literal_eval
from collections import defaultdict
import pickle
from statistics import mean
import yaml
import os

from evaluate import load
import pandas as pd
import torch
from tqdm import tqdm
from transformers import logging
from sentence_transformers import SentenceTransformer

from reranker import BGEReranker
from gnd_dataset import GNDDataset
from gnd_graph import GNDGraph
from utils import recall_at_k, precision_at_k, f1_at_k, jaccard_similarity, weighted_precision, SEP_TOKEN



logging.set_verbosity_error()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Evaluate the model on the GND dataset.")
parser.add_argument("--reranker_model", type=str, help="Name of the reranker model.", default="BAAI/bge-reranker-v2-m3")
parser.add_argument("--predictions_file", type=str, help="Path to the predictions file.")
parser.add_argument("--write_reranked", type=bool, default=True, help="Whether to write the reranked predictions to the given file.")
parser.add_argument("--config", type=str, help="Path to the configuration file.")

arguments = parser.parse_args()
reranker_str = arguments.reranker_model
pred_file = arguments.predictions_file
write_reranked = arguments.write_reranked
config_path = arguments.config

# Load config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

reranker = BGEReranker(reranker_str, device=DEVICE)

# Load GND graph
gnd_path = config["graph_path"]
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

# Load predictions file.
test_df = pd.read_csv(pred_file)

dataset_path = config["dataset_path"]
ds = GNDDataset(
    data_dir=dataset_path,
    gnd_graph=gnd_graph,
    config=config,
    load_from_disk=True
)

all_metrics_dict = {}

rec_all = []
prec_all = []
f1_all = []
jaccard = []
weighted_prec = []

# Convert to list
test_df["predictions"] = test_df["predictions"].apply(literal_eval)
test_df["label-ids"] = test_df["label-ids"].apply(literal_eval)

label_names = []
for ids in test_df["label-ids"]:
    instance_names = []
    for i in ids:
        if i in gnd_graph.nodes:
            i_pref = gnd_graph.pref_label_name(i)
            instance_names.append(i_pref)
    label_names.append(instance_names)
test_df["label-names"] = label_names

for preds_i, golds_i in test_df[["predictions", "label-ids"]].itertuples(index=False):
    rec_all.append(recall_at_k(y_pred=preds_i, y_true=golds_i))
    prec_all.append(precision_at_k(y_pred=preds_i, y_true=golds_i))
    f1_all.append(f1_at_k(y_pred=preds_i, y_true=golds_i))
    jaccard.append(jaccard_similarity(y_true=golds_i, y_pred=preds_i))
    weighted_prec.append(weighted_precision(y_pred=preds_i, y_true=golds_i, graph=gnd_graph))

all_metrics_dict["recall"] = mean(rec_all)
all_metrics_dict["precision"] = mean(prec_all)
all_metrics_dict["f1"] = mean(f1_all)
all_metrics_dict["jaccard"] = mean(jaccard)
all_metrics_dict["weighted_precision"] = mean(weighted_prec)

print("Metrics without reranking:")
print(f'Recall: {all_metrics_dict["recall"] }\nPrecision: {all_metrics_dict["precision"]}\nF1: {all_metrics_dict["f1"]}')
print(f"Jaccard Similarity: {all_metrics_dict['jaccard']}")
print(f"Weighted Precision: {all_metrics_dict['weighted_precision']}")

if "reranked-predictions" not in test_df.columns:
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

recall_dict = {}
precision_dict = {}
f1_dict = {}
ks = [1, 3, 5, 10]

for preds_i, golds_i in tqdm(test_df[["reranked-predictions", "label-ids"]].itertuples(index=False)):
    for k in ks:
        if k not in recall_dict:
            recall_dict[k] = []
            precision_dict[k] = []
            f1_dict[k] = []
        recall_dict[k].append(recall_at_k(y_pred=preds_i, y_true=golds_i, k=k))
        precision_dict[k].append(precision_at_k(y_pred=preds_i, y_true=golds_i, k=k))
        f1_dict[k].append(f1_at_k(y_pred=preds_i, y_true=golds_i, k=k))

print("Metrics with reranking:")
for k in ks:
    rec_k  = f"recall@{k}"
    prec_k = f"precision@{k}"
    f1_k   = f"f1@{k}"
    all_metrics_dict[rec_k] = mean(recall_dict[k])
    all_metrics_dict[prec_k] = mean(precision_dict[k])
    all_metrics_dict[f1_k] = mean(f1_dict[k])
    print(f'Recall@{k}: {all_metrics_dict[rec_k]}')
    print(f'Precision@{k}: {all_metrics_dict[prec_k]}')
    print(f"F1@{k}: {all_metrics_dict[f1_k]}")
    print("-----------------")

## Labelwise metrics
def get_dict():
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
macro_metrics = defaultdict(get_dict)

label_freq = ds.label_frequency(
    ds["train"]["label-ids"],
)

for preds_i, golds_i in tqdm(test_df[["reranked-predictions", "label-ids"]].itertuples(index=False)):
    correct = set(preds_i).intersection(set(golds_i))
    for pred in preds_i:
        if pred in correct:
            macro_metrics[pred]["tp"] += 1
        else:
            macro_metrics[pred]["fp"] += 1

    for gold in golds_i:
        if gold not in correct:
            macro_metrics[gold]["fn"] += 1

bins = [
    (-float("inf"), 1),
    (1, 2),
    (2, 11),
    (11, 101),
    (101, 1001),
    (1001,  float("inf"))
]
label_freq_grouped_prec = { end-1: [] for _, end in bins }
label_freq_grouped_rec = { end-1: [] for _, end in bins }
support_lf = {end-1: 0 for _, end in bins }
entity_types = {}

for label, values in macro_metrics.items():
    tp = values["tp"]
    fp = values["fp"]
    fn = values["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    type_i = gnd_graph.node_type(label) 
    if type_i not in entity_types:
        entity_types[type_i] = {"precision": [], "recall": []}
    entity_types[type_i]["precision"].append(precision)
    entity_types[type_i]["recall"].append(recall)
    freq = label_freq.get(label, 0)
    for start, end in bins:
        if freq >= start and freq < end:
            freq_key = end - 1
            label_freq_grouped_prec[freq_key].append(precision)
            label_freq_grouped_rec[freq_key].append(recall)
            support_lf[freq_key] += 1
            break

# Distribution of frequency of labels in test set
label_freq_test = {end-1: set() for _, end in bins }
for label in test_df["label-ids"]:
    for label_i in label:
        freq = label_freq.get(label_i, 0)
        for start, end in bins:
            if freq >= start and freq < end:
                freq_key = end - 1
                label_freq_test[freq_key].add(label_i)
                break

print("Distribution of label frequencies in test set:")
for key in label_freq_test.keys():
    label_freq_test[key] = len(label_freq_test[key])
    print(f"Label frequency <= {key}: {label_freq_test[key]}")

print("-----------------")
print("Performance per label frequency:")

label_frequencies_mean = {}
for key in label_freq_grouped_prec.keys():
    current_prec = label_freq_grouped_prec[key]
    current_rec = label_freq_grouped_rec[key]

    if current_prec:
        macro_prec = mean(current_prec)
    else:
        macro_prec = 0.0
    if current_rec:
        macro_rec = mean(current_rec)
    else:
        macro_rec = 0.0
    label_frequencies_mean[key] = {
        "precision": macro_prec,
        "recall": macro_rec,
        "support": label_freq_test[key],
    }
    print(f"Label frequency <= {key}: Precision: {macro_prec}, Recall: {macro_rec}")
    print(f"Support: {label_freq_test[key]}")

all_metrics_dict["label_frequencies"] = label_frequencies_mean
print("-----------------")

print("Performance per entity type:")
entity_types_mean = {}
for entity_type, values in entity_types.items():
    macro_prec = mean(values["precision"])
    macro_rec = mean(values["recall"])
    entity_types_mean[entity_type] = {
        "precision": macro_prec,
        "recall": macro_rec,
    }
    print(f"{entity_type}: Precision: {macro_prec}, Recall: {macro_rec}")

all_metrics_dict["entity_types"] = entity_types_mean

print("-----------------")

# Distribution of entity types in predictions
entity_types_pred = {}
entity_types_gold = {}
avg_no_preds = 0
for preds_i, golds_i in zip(test_df["reranked-predictions"], test_df["label-ids"]):
    avg_no_preds += len(preds_i)
    for pred in preds_i:
        type_i = gnd_graph.node_type(pred)
        if type_i not in entity_types_pred:
            entity_types_pred[type_i] = 0
        entity_types_pred[type_i] += 1
    for gold in golds_i:
        type_i = gnd_graph.node_type(pred)
        if type_i not in entity_types_gold:
            entity_types_gold[type_i] = 0
        entity_types_gold[type_i] += 1

print("Distribution of entity types in predictions:")
for entity_type, count in entity_types_pred.items():
    print(f"{entity_type}: {count}")

print("-----------------")

print("Distribution of entity types in gold labels:")
for entity_type, count in entity_types_gold.items():
    print(f"{entity_type}: {count}")

print("-----------------")

avg_no_preds /= len(test_df)
print(f"Average number of predictions per sample: {avg_no_preds}")
all_metrics_dict["avg_no_preds"] = avg_no_preds

# MT Metrics
bertscore = load("bertscore")
meteor = load('meteor')
rouge = load('rouge')

gold_labels = test_df["label-names"].apply(lambda x: f"{SEP_TOKEN} ".join(x))

if "raw_predictions" in test_df.columns:
    raw_preds = test_df["raw_predictions"]

    bert_results = bertscore.compute(
        predictions=raw_preds, 
        references=gold_labels, 
        model_type="bert-base-multilingual-cased", 
        lang="de"
    )
    bert_results = {
        "precision": mean(bert_results["precision"]),
        "recall": mean(bert_results["recall"]),
        "f1": mean(bert_results["f1"])
    }
    print(f"BERTScore: {bert_results}")
    all_metrics_dict["bertscore"] = bert_results

    meteor_results = meteor.compute(
        predictions=raw_preds, 
        references=gold_labels
    )
    print(f"Meteor: {meteor_results}")
    all_metrics_dict["meteor"] = float(meteor_results['meteor'])

    rouge_results = rouge.compute(
        predictions=raw_preds, 
        references=gold_labels
    )
    print(f"Rouge: {rouge_results}")
    all_metrics_dict["rouge"] = {k: float(v) for k, v in rouge_results.items()}

# Group Labels by their similarity to the title.
sent_model = SentenceTransformer(config["sentence_transformer_model"])
tokenizer = sent_model.tokenizer

idnl2index = {}
index2idn = {}
i = 0
for idns in tqdm(test_df["label-ids"]):
    for idn in idns:
        if idn not in idnl2index:
            idnl2index[idn] = i
            index2idn[i] = idn
            i += 1

emb_title =  sent_model.encode(test_df["title"], batch_size=256, show_progress_bar=True)
gold_strings = [gnd_graph.pref_label_name(idn) for idn in idnl2index.keys()]
emb_gold = sent_model.encode(gold_strings, batch_size=256, show_progress_bar=True)

similarity_dict = {}

for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
    gold_labels = row["label-ids"]
    for g in gold_labels:
        g_i = idnl2index[g]
        sim = sent_model.similarity(emb_title[i], emb_gold[g_i])
        correct = g in row["predictions"]
        similarity_dict[(row["title"], g)] = {
            "similarity": sim,
            "correct": correct,
        }

grouped_sim = {
    "not-similar": [],
    "somewhat-similar": [],
    "similar": [],
    "very-similar": [],
}

for (title, label_name), value in similarity_dict.items():
    sim = value["similarity"]
    correct = value["correct"]
    if sim < 0.25:
        grouped_sim["not-similar"].append(correct)
    elif sim < 0.5:
        grouped_sim["somewhat-similar"].append(correct)
    elif sim < 0.75:
        grouped_sim["similar"].append(correct)
    else:
        grouped_sim["very-similar"].append(correct)

for group, values in grouped_sim.items():
    grouped_sim[group] = mean(values) if values else 0.0

print("Performance grouped by similarity to title:")
for group, value in grouped_sim.items():
    print(f"{group}: {value}")
all_metrics_dict["similarity_to_title"] = grouped_sim

# Save all metrics to a YAML file
file_name = os.path.basename(pred_file).split(".")[0]
eval_path = os.path.join(os.path.dirname(pred_file), f"{file_name}_eval.yaml")
with open(eval_path, "w") as f:
    yaml.dump(all_metrics_dict, f, indent=2, sort_keys=False, allow_unicode=True)

