import pandas as pd

def add_label_freq_info(df, freq_dict):
    df["label-freq"] = df["label-id"].apply(
        lambda x: freq_dict.get(x, 0)
        )

    df["label-freq"] = pd.cut(
        df["label-freq"], 
        bins=[-1, 0, 1, 10, 100, 1000, 100_000_000],
        labels=["x=0", "x=1", "1<x<=10", "10<x<=100", "100<x<=1000", "x>1000"]
        )
    return df