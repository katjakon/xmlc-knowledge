import argparse
import os
from ast import literal_eval
from statistics import mean
import pickle
import yaml
from functools import reduce

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from tqdm import tqdm

from data_collator import DataCollator
from default_config import default_config
from gnd_dataset import GNDDataset
from gnd_graph import GNDGraph

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config = default_config
retriever_model =  "KatjaK/gnd_retriever_full"
map_model = "BAAI/bge-m3"
res_dir = "results"
pred_files = [
    "hard-prompting-3b-context-label-5-ft",
    "few-shot-3k-title-retrieval-3b",
    ]
gnd_path  = config["graph_path"]
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

n_ex = [100]# [3596, 8273, 245, 7691, 6005]
cols = ["reranked-predictions", "label-ids", "scores"]

def get_eval_dfs(exp_dir, indices, k=3):
    exp_path = os.path.join(res_dir, exp_dir)
    exp_files = [os.path.join(exp_path, exp) for exp in os.listdir(exp_path) if exp.endswith(".csv")]
    top_k_dict = {}
    for exp in exp_files:
        test_df = pd.read_csv(exp)
        for c in cols:
            test_df[c] = test_df[c].apply(literal_eval)
        sampled = test_df.iloc[indices]
        for idx, record in sampled.iterrows():
            doc_idn = record["doc_idn"]
            if doc_idn not in top_k_dict:
                top_k_dict[doc_idn] = {}
            for pred, score in zip(record["reranked-predictions"], record["scores"]):
                top_k_dict[doc_idn][pred] = score
    for doc in top_k_dict:
        top_pred = sorted(top_k_dict[doc], key=lambda x: top_k_dict[doc][x], reverse=True)[:k]
        top_k_dict[doc] = {idn: top_k_dict[doc][idn] for idn in top_pred}
    return [pd.DataFrame.from_dict(top_k_dict[doc], orient="index") for doc in top_k_dict]

for exp in pred_files:
    print(get_eval_dfs(exp, indices=[100]))  

exit()

exps_evals = []
for experiment in pred_files:
    experiment = os.path.join(experiment)
    exp_name = os.path.split(experiment)[-2]
    score_col = exp_name
    eval_df = {
    "id": [],
    score_col: []
    }    
    test_df = pd.read_csv(experiment)
    test_df["predictions"] = test_df["predictions"].apply(literal_eval)
    test_df["label-ids"] = test_df["label-ids"].apply(literal_eval)
    test_df["reranked-predictions"] = test_df["reranked-predictions"].apply(literal_eval)
    test_df["scores"] = test_df["scores"].apply(literal_eval)
    samples = test_df.iloc[n_ex]
    for idx, sample in samples.iterrows():
        predictions = sample["reranked-predictions"]
        scores = sample["scores"]
        gold = sample["label-ids"]
        for pred, score in zip(predictions, scores):
            eval_df[score_col].append(score)
            eval_df["id"].append(pred)
        false_negatives = set(gold).difference(set(predictions))
        for fn in false_negatives:
            eval_df[score_col].append(0.0)
            eval_df["id"].append(fn)
        eval_df = pd.DataFrame.from_dict(eval_df)
        exps_evals.append(eval_df)
    

df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['id'],
                                            how='outer'), exps_evals)

df_merged["label"] = df_merged["id"].apply(lambda x: gnd_graph.pref_label_name(x))
df_merged["gold"] = df_merged["id"].apply(lambda x: x in samples["label-ids"].to_list()[0])

print(samples["title"])
print(df_merged.sort_values(by="gold"))