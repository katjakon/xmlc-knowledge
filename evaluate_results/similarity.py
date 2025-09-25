import pandas as pd

def get_similarity_data(sentence_model, data, gnd_graph, batch_size=256):
    label_strings, idx2idn = gnd_graph.mapping()
    idn2idx = {value: key for key, value in idx2idn.items()}
    title_embed = sentence_model.encode(data["title"], batch_size=batch_size, show_progress_bar=True)
    label_embeddings = sentence_model.encode(label_strings,  batch_size=batch_size, show_progress_bar=True)
    return {
        "title_embeddings": title_embed, 
        "label_embeddings": label_embeddings,
        "idn2idx": idn2idx,
        "idx2idn": idx2idn
    }

def bin_similarity(df):
    df["similarity"] = pd.cut(
    df["similarity"], 
    bins=[0, 0.25, 0.5, 0.75, 1],
    labels=["not similar", "somewhat similar", "similar", "very similar"]
    )
    return df
