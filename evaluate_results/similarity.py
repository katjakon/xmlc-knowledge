from tqdm import tqdm

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


# idnl2index = {}
# index2idn = {}
# i = 0
# for idns in tqdm(test_df["label-ids"]):
#     for idn in idns:
#         if idn not in idnl2index:
#             idnl2index[idn] = i
#             index2idn[i] = idn
#             i += 1

# emb_title =  sent_model.encode(test_df["title"], batch_size=256, show_progress_bar=True)
# gold_strings = [gnd_graph.pref_label_name(idn) for idn in idnl2index.keys()]
# emb_gold = sent_model.encode(gold_strings, batch_size=256, show_progress_bar=True)

# similarity_dict = {}

# for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
#     gold_labels = row["label-ids"]
#     for g in gold_labels:
#         g_i = idnl2index[g]
#         sim = sent_model.similarity(emb_title[i], emb_gold[g_i])
#         correct = g in row["predictions"]
#         similarity_dict[(row["title"], g)] = {
#             "similarity": sim,
#             "correct": correct,
#         }

# grouped_sim = {
#     "not-similar": [],
#     "somewhat-similar": [],
#     "similar": [],
#     "very-similar": [],
# }

# for (title, label_name), value in similarity_dict.items():
#     sim = value["similarity"]
#     correct = value["correct"]
#     if sim < 0.25:
#         grouped_sim["not-similar"].append(correct)
#     elif sim < 0.5:
#         grouped_sim["somewhat-similar"].append(correct)
#     elif sim < 0.75:
#         grouped_sim["similar"].append(correct)
#     else:
#         grouped_sim["very-similar"].append(correct)

# for group, values in grouped_sim.items():
#     grouped_sim[group] = mean(values) if values else 0.0

# print("Performance grouped by similarity to title:")
# for group, value in grouped_sim.items():
#     print(f"{group}: {value}")