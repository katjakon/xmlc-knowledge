## Plot evaluation metrics. 
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yaml

MT_METRCICS = [
    "meteor",
    "bertscore",
    "rouge"
]
MAPPED_METRICS = [
    "precision",
    # "precision@3",
    # "precision@5",
    "recall",
    # "recall@3",
    # "recall@5",
    "f1",
    # "f1@3",
    # "f1@5",
    # "jaccard",
    "weighted_precision",

]

lf_key = "label_frequencies"
entity_key = "entity_types"
similarity_key = "similarity_to_title"

eval_dir = [
    # "results/retrieval-ft-no-neighbors",
    # "results/retrieval-no-neighbors",
    # "results/finetuned-retriever-partial",
    # "results/hard-prompting-context-label-5",
    # "results/ft-hard-prompting-context-label-5",
    "results/prompt-tuning-baseline-full",
    # "results/pt-txt-context-3-label",
    "results/prompt-tuning-from_gnd",
    # "results/pt-from_gnd_w_alt",
    "results/prompt-tuning-baseline-small",
    "results/pt-from_gnd_small",
    # "results/pt-from_gnd_fused",
    # "results/few-shot-baseline",
    # "results/hard-prompting-baseline"


]

hue = "experiment"  # graph-based
dir = "plots_pt"

def load_eval_data(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_eval_metrics(eval_file, metrics_names=None):
    metrics = []
    data = load_eval_data(eval_file)
    for key, value in data.items():
        if metrics_names and key not in metrics_names:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                metrics.append({
                    'metric': f"{key}.{sub_key}",
                    'value': sub_value,
                    'experiment': eval_file.split('/')[-2],
                    "graph-based": "hop" in eval_file.split('/')[-2]
                })
        else:
            metrics.append({
                'metric': key,
                'value': value,
                'experiment': eval_file.split('/')[-2],
                "graph-based": "hop" in eval_file.split('/')[-2]
            })
    return metrics

def get_lf_frequencies(eval_file):
    lf_data = []
    data = load_eval_data(eval_file)
    if lf_key in data:
        for freq, metrics_dict in data[lf_key].items():
            lf_data.append({
                'frequency': freq,
                'precision': metrics_dict.get('precision', 0),
                'recall': metrics_dict.get('recall', 0),
                'experiment': eval_file.split('/')[-2],
                "graph-based": "hop" in eval_file.split('/')[-2],
                "name": "label frequency"
            })
    return lf_data

def get_entity_types(eval_file):
    entity_data = []
    data = load_eval_data(eval_file)
    if entity_key in data:
        for entity, metrics_dict in data[entity_key].items():
            entity_data.append({
                'entity': entity,
                'precision': metrics_dict.get('precision', 0),
                'recall': metrics_dict.get('recall', 0),
                'experiment': eval_file.split('/')[-2],
                "graph-based": "hop" in eval_file.split('/')[-2],
                "name": "entity type"
            })
    return entity_data

def get_similarity_to_title(eval_file):
    similarity_data = []
    data = load_eval_data(eval_file)
    if similarity_key in data:
        for similarity, accuracy in data[similarity_key].items():
            similarity_data.append({
                'similariy': similarity,
                'accuracy': accuracy,
                'experiment': eval_file.split('/')[-2],
                "graph-based": "hop" in eval_file.split('/')[-2],
                "name": "similarity to title"
            })
    return similarity_data

def plot(df, x, y, hue=None, title=None, xlabel=None, ylabel=None, dir=None):
    plt.figure(figsize=(16, 8))
    sns.barplot(x=x, y=y, hue=hue, data=df, errorbar="sd")
    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)
    plt.xticks(rotation=90)
    if title:
        plt.title(title)
    if hue:
        plt.legend(title=hue.capitalize())
    plt.tight_layout()
    plt.show()
    file_name = title.replace(" ", "_").lower() if title else "plot"
    if dir:
        os.makedirs(dir, exist_ok=True)
        file_name = os.path.join(dir, file_name)
    plt.savefig(f'{file_name}.png', dpi=300)

if __name__ == "__main__":
    all_exact_metrics = []
    all_mt_metrics = []
    all_lf_data = []
    all_entity_data = []
    all_similarity_data = []

    for eval_dir_path in eval_dir:
        eval_files = [os.path.join(eval_dir_path, f) for f in os.listdir(eval_dir_path) if f.endswith('.yaml')]
        for eval_file in eval_files:
            all_exact_metrics.extend(get_eval_metrics(eval_file, metrics_names=MAPPED_METRICS))
            all_mt_metrics.extend(get_eval_metrics(eval_file, metrics_names=MT_METRCICS))
            all_lf_data.extend(get_lf_frequencies(eval_file))
            all_entity_data.extend(get_entity_types(eval_file))
            all_similarity_data.extend(get_similarity_to_title(eval_file))
    if all_exact_metrics:
        df_metrics = pd.DataFrame(all_exact_metrics)
        plot(
            df=df_metrics,
            x='metric',
            y='value',
            hue=hue,
            title='Mapped Match Metrics Comparison',
            xlabel='Metrics',
            ylabel='Values',
            dir=dir
        )
    if all_mt_metrics:
        df_mt = pd.DataFrame(all_mt_metrics)
        plot(
            df=df_mt,
            x='metric',
            y='value',
            hue=hue,
            title='Machine Translation Metrics Comparison',
            xlabel='Metrics',
            ylabel='Values',
            dir=dir
        )
    if all_lf_data:
        df_lf = pd.DataFrame(all_lf_data)
        plot(
            df=df_lf,
            x='frequency',
            y='precision',
            hue=hue,
            title='Label Frequencies Comparison',
            xlabel='Label Frequency',
            ylabel='Precision',
            dir=dir
        )
    if all_entity_data:
        df_entity = pd.DataFrame(all_entity_data)
        plot(
            df=df_entity,
            x='entity',
            y='precision',
            hue=hue,
            title='Entity Types Comparison',
            xlabel='Entity Type',
            ylabel='Precision',
            dir=dir
        )
    if all_similarity_data:
        df_similarity = pd.DataFrame(all_similarity_data)
        plot(
            df=df_similarity,
            x='similariy',
            y='accuracy',
            hue=hue,
            title='Similarity to Title Comparison',
            xlabel='Similarity',
            ylabel='Accuracy',
            dir=dir
        )
