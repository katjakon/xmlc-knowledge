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
    # "precision",
    # "recall",
    # "f1",
    "precision@1",
    "precision@3",
    "precision@5",
    "recall@1",
    "recall@3",
    "recall@5",
    # "f1@1",
    # "f1@3",
    # "f1@5",
    # "jaccard",
    "weighted_precision",

]

lf_key = "label_frequencies"
entity_key = "entity_types"
similarity_key = "similarity"
genres_key = "genres"

eval_dir = [
    "results/hard-prompting-baseline",
    "results/hard-prompting-3b-context-label-3k-1h-ft",
    "results/few-shot-ft-3k-1hop-label-retrieval-3b",
    "results/few-shot-3k-title-retrieval-3b",

]

exp_name_mapping = {
    "results/hard-prompting-baseline": "Input-only",
    "results/hard-prompting-3b-context-label-3k-1h-ft": "Label-only with 3 labels & 1-hop",
    "results/few-shot-ft-3k-1hop-label-retrieval-3b": "Label-based, few-shot with 3 labels & 1-hop",
    "results/few-shot-3k-title-retrieval-3b": "Title-based, few-shot with 3 instances"
}

entity_mapping = {
    "Geografikum": "Geographic Entity", 
    "Konferenz": "Conference", 
    "Körperschaft": "Corporate Body",
    "Person (individualisiert)": "Person",
    "Sachbegriff": "Topic",
    "Werk": "Work"
}

hue = "experiment"  # graph-based
dir = "plots_pt"

def load_eval_data(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_eval_metrics(eval_file, metrics_names=None):
    metrics = []
    data = load_eval_data(eval_file)
    raw_name = os.path.split(eval_file)[-2]
    exp_name = exp_name_mapping.get(raw_name, raw_name)
    for key, value in data.items():
        if metrics_names and key not in metrics_names:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                metrics.append({
                    'metric': f"{key}.{sub_key}",
                    'value': sub_value,
                    'experiment': exp_name,
                    "graph-based": "hop" in raw_name
                })
        else:
            metrics.append({
                'metric': key,
                'value': value,
                'experiment': exp_name,
                "graph-based": "hop" in raw_name
            })
    return metrics

def get_lf_frequencies(eval_file):
    lf_data = []
    data = load_eval_data(eval_file)
    raw_name = os.path.split(eval_file)[-2]
    exp_name = exp_name_mapping.get(raw_name, raw_name)
    if lf_key in data:
        for freq, metrics_dict in data[lf_key].items():
            lf_data.append({
                'frequency': str(freq).lower(),
                'precision': metrics_dict.get('precision', 0),
                'recall': metrics_dict.get('recall', 0),
                'f1': metrics_dict.get('f1', 0),
                'experiment': exp_name,
                "graph-based": "hop" in  raw_name,
                "name": "label frequency"
            })
    return lf_data

def get_entity_types(eval_file):
    entity_data = []
    data = load_eval_data(eval_file)
    raw_name = os.path.split(eval_file)[-2]
    exp_name = exp_name_mapping.get(raw_name, raw_name)
    if entity_key in data:
        for entity, metrics_dict in data[entity_key].items():
            entity_data.append({
                'entity': entity_mapping.get(entity),
                'precision': metrics_dict.get('precision', 0),
                'recall': metrics_dict.get('recall', 0),
                'f1': metrics_dict.get('f1', 0),
                'experiment': exp_name,
                "graph-based": "hop" in raw_name,
                "name": "entity type"
            })
    return entity_data

def get_similarity_to_title(eval_file):
    similarity_data = []
    data = load_eval_data(eval_file)
    raw_name = os.path.split(eval_file)[-2]
    exp_name = exp_name_mapping.get(raw_name, raw_name)
    if similarity_key in data:
        for similarity, metrics_dict in data[similarity_key].items():
            similarity_data.append({
                'similarity': similarity.capitalize(),
                'precision': metrics_dict.get('precision', 0),
                'recall': metrics_dict.get('recall', 0),
                'f1': metrics_dict.get('f1', 0),
                'experiment': exp_name,
                "graph-based": raw_name,
                "name": "similarity to title"
            })
    return similarity_data

def get_genres(eval_file):
    genre_data = []
    data = load_eval_data(eval_file)
    raw_name = os.path.split(eval_file)[-2]
    exp_name = exp_name_mapping.get(raw_name, raw_name)
    if genres_key in data:
        for genre, metrics_dict in data[genres_key].items():
            genre_data.append({
                'genre': genre.capitalize(),
                'precision': metrics_dict.get('precision', 0),
                'recall': metrics_dict.get('recall', 0),
                'f1': metrics_dict.get('f1', 0),
                'experiment': exp_name,
                "graph-based": "hop" in raw_name,
                "name": "genre"
            })
    return genre_data

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
    #plt.show()
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
    all_genre_data = []

    for eval_dir_path in eval_dir:
        eval_files = [os.path.join(eval_dir_path, f) for f in os.listdir(eval_dir_path) if f.endswith('.yaml')]
        for eval_file in eval_files:
            all_exact_metrics.extend(get_eval_metrics(eval_file, metrics_names=MAPPED_METRICS))
            all_mt_metrics.extend(get_eval_metrics(eval_file, metrics_names=MT_METRCICS))
            all_lf_data.extend(get_lf_frequencies(eval_file))
            all_entity_data.extend(get_entity_types(eval_file))
            all_similarity_data.extend(get_similarity_to_title(eval_file))
            all_genre_data.extend(get_genres(eval_file))
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
            x='similarity',
            y='precision',
            hue=hue,
            title='Similarity to Title Comparison',
            xlabel='Similarity',
            ylabel='Precision',
            dir=dir
        )
    if all_genre_data:
        df_genre = pd.DataFrame(all_genre_data)
        plot(
            df=df_genre,
            x='genre',
            y='precision',
            hue=hue,
            title='Performance for Genres',
            xlabel='Genre',
            ylabel='Precision',
            dir=dir
        )