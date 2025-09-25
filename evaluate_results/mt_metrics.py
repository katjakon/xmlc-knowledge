from statistics import mean

from evaluate import load

from utils import SEP_TOKEN

def gold_label_strings(df, graph):
    label_list = df["label-ids"].apply(
        lambda x: [graph.pref_label_name(idn) for idn in x if idn in graph]
    )
    gold_labels = label_list.apply(lambda x: f"{SEP_TOKEN} ".join(x))
    return gold_labels

def bertscore_results(pred_strings, gold_strings): 
    bertscore = load("bertscore") 
    bert_results = bertscore.compute(
        predictions=pred_strings, 
        references=gold_strings, 
        model_type="bert-base-multilingual-cased", 
        lang="de"
    )
    bert_results = {
    "precision": mean(bert_results["precision"]),
    "recall": mean(bert_results["recall"]),
    "f1": mean(bert_results["f1"]) }
    return bert_results

def meteor_results(pred_strings, gold_strings):
    meteor = load('meteor')
    results = meteor.compute(
        predictions=pred_strings, 
        references=gold_strings
    )
    return float(results['meteor'])

def rouge_results(pred_strings, gold_strings):
    rouge = load('rouge')
    results = rouge.compute(
        predictions=pred_strings, 
        references=gold_strings
    )
    return {k: float(v) for k, v in results.items()}

    