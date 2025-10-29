import pandas as pd
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt


SHORT_MAPPING = {
    "0": "General, computer Science, information Science",
    "1": "Philosophy and psychology",
    "2": "Religion",
    "3": "Social sciences",
    "4": "Language",
    "5": "Natural sciences and mathematics",
    "6": "Technology, medicine, applied sciences",
    "7": "Arts and entertainment",
    "8": "Literature",
    "9": "History and geography",
    "B": "Fiction",
    "S": "Schoolbooks",
    "K": "Children's and young adult literature"
}

def add_hsg_info(df, docid2hsg, use_label=True, hsg2label=None):
    df["hsg"] = df["doc_idn"].apply(lambda x: docid2hsg.get(x))
    if use_label:
        if hsg2label is None:
            raise ValueError("Need mapping from code to label!")
        df["hsg"] = df["hsg"].apply(lambda x: hsg2label.get(x))
    return df

def hsg_data(path, shorten_codes=True):
    df = pd.read_csv(path)
    if shorten_codes:
        df["hsg"] = df["hsg"].str[:1]
        hsg2label = SHORT_MAPPING
    else:
        hsg2label = {hsg_code: hsg_label for hsg_code, hsg_label in zip(df["hsg"], df["hsg_label"])}
    docid2hsg = {doc_id: hsg_code for doc_id, hsg_code in zip(df["doc_id"], df["hsg"])}
    return docid2hsg, hsg2label
