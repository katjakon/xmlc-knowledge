import argparse
from ast import literal_eval
import pickle
from statistics import mean

from evaluate import load
import pandas as pd
import torch
from tqdm import tqdm
from transformers import logging

from reranker import BGEReranker
from utils import recall_at_k, precision_at_k, f1_at_k, jaccard_similarity, SEP_TOKEN


logging.set_verbosity_error()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Evaluate the model on the GND dataset.")
parser.add_argument("--gnd_graph", type=str, help="Path to the GND graph file (pickle).")
parser.add_argument("--reranker_model", type=str, help="Name of the reranker model.", default="BAAI/bge-reranker-v2-m3")
parser.add_argument("--predictions_file", type=str, help="Path to the predictions file.")
parser.add_argument("--write_reranked", type=bool, default=True, help="Whether to write the reranked predictions to the given file.")

arguments = parser.parse_args()
reranker_str = arguments.reranker_model
gnd_path = arguments.gnd_graph 
pred_file = arguments.predictions_file
write_reranked = arguments.write_reranked

gnd = pickle.load(open(gnd_path, "rb"))
test_df = pd.read_csv(pred_file)


rec_all = []
prec_all = []
f1_all = []
jaccard = []

# Convert to list
test_df["predictions"] = test_df["predictions"].apply(literal_eval)
test_df["label-ids"] = test_df["label-ids"].apply(literal_eval)
test_df["label-names"] = test_df["label-names"].apply(literal_eval)

for preds_i, golds_i in test_df[["predictions", "label-ids"]].itertuples(index=False):
    rec_all.append(recall_at_k(y_pred=preds_i, y_true=golds_i))
    prec_all.append(precision_at_k(y_pred=preds_i, y_true=golds_i))
    f1_all.append(f1_at_k(y_pred=preds_i, y_true=golds_i))
    jaccard.append(jaccard_similarity(y_true=golds_i, y_pred=preds_i))

print("Metrics without reranking:")
print(f"Recall: {mean(rec_all)}\nPrecision: {mean(prec_all)}\nF1: {mean(f1_all)}")
print(f"Jaccard Similarity: {mean(jaccard)}")

if "reranked-predictions" not in test_df.columns:
    reranker = BGEReranker(reranker_str, device=DEVICE)
    test_df = reranker.rerank(
        test_df,
        gnd,
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
    print(f"Recall@{k}: {mean(recall_dict[k])}")
    print(f"Precision@{k}: {mean(precision_dict[k])}")
    print(f"F1@{k}: {mean(f1_dict[k])}")
    print("-----------------")


test_df["label-string"] = test_df["label-names"].str.join(SEP_TOKEN)
bertscore = load("bertscore")
predictions = test_df["raw_predictions"][:100]
references = test_df["label-string"][:100]
results = bertscore.compute(predictions=predictions, references=references, model_type="bert-base-multilingual-cased")

print("BERT SCORE")
print(f"Recall: {mean(results['recall'])}")
print(f"Precision: {mean(results['precision'])}")
print(f"F1: {mean(results['f1'])}")

