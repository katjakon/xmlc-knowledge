import pandas as pd
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt

path = "gnd/hsg-mapping-small.csv"
pred_file = "results/few-shot-baseline-8b/predictions-test-few-shot-seed-42.csv"
df = pd.read_csv(path)
df["hsg"] = df["hsg"].str[:1]

docid2hsg = {
    doc_id: hsg_code for doc_id, hsg_code in zip(df["doc_id"], df["hsg"])
}

mapping = {
    "0": "General, Computer Science, Information Science",
    "1": "Philosophy and Psychology",
    "2": "Religion",
    "3": "Social Sciences",
    "4": "Language",
    "5": "Natural sciences and mathematics",
    "6": "Technology, medicine, applied sciences",
    "7": "Arts and entertainment",
    "8": "Literature",
    "9": "History and Geography",
    "B": "Fiction",
    "S": "Schoolbooks",
    "K": "Children's and Young Adult Literature"
}

test_df = pd.read_csv(pred_file)
test_df["predictions"] = test_df["predictions"].apply(literal_eval)
test_df["label-ids"] = test_df["label-ids"].apply(literal_eval)

test_df["hsg"] = test_df["doc_idn"].apply(lambda x: docid2hsg.get(x))

test_df["correct"] = [
    len(set(gold) & set(pred))
    for gold, pred in zip(test_df["label-ids"], test_df["predictions"])
]
test_df["n_gold"] = test_df["label-ids"].str.len()
test_df["n_pred"] = test_df["predictions"].str.len()
test_df["recall"] = test_df["correct"] / test_df["n_gold"]
test_df["precision"] = test_df["correct"] / test_df["n_pred"]
test_df["f1"] = 2 * (test_df["precision"]*test_df["recall"]) / (test_df["precision"]+test_df["recall"])

result = test_df.groupby("hsg")["f1"].mean()

result.index = [mapping[i] for i in result.index]
print(result)

plt.figure(figsize=(16, 8))
sns.barplot(x=result.index, y=result, errorbar="sd")
plt.xlabel("Categories")
plt.ylabel("F1 Score")
plt.title("F1 Score per Subject Categories")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('subject_categories.png', dpi=300)

