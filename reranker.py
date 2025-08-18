
from datasets import Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BGEReranker:

    def __init__(self, model_path, device=None):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if device is not None:
            self.model.to(device)
        self.device = device

    def similarities(self, input_ids, attention_masks=None):
        """
        input_ids: torch.Tensor of shape (batch_size, seq_len)
        attention_masks: torch.Tensor of shape (batch_size, seq_len)

        Returns:
        torch.Tensor of shape (batch_size, )
        """
        self.model.eval()
        if attention_masks is None:
            attention_masks = torch.ones_like(input_ids)
        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        with torch.no_grad():
            scores = self.model(
                input_ids=input_ids, 
                attention_mask=attention_masks,
                return_dict=True).logits.view(-1, ).float()
            scores = torch.sigmoid(scores)  # Apply sigmoid to get probabilities
        return scores
    
    def tokenize(self, pair, max_len=128):
        return self.tokenizer(pair, padding=True, truncation=True, return_tensors='pt', max_length=max_len)

    def create_rerank_dataset(self, data_frame, gnd):
        """
        Creates a rerank dataset from the given DataFrame.

        Args:
            data_frame (pd.DataFrame): The DataFrame containing the data. Needs columns 'predictions', 'title'.
            gnd: The GND graph (networkx graph) used to get the label strings.

        Returns:
            Dataset: The rerank dataset.
        """
        pair_dict = {
            "pair": [],
            "label-ids": [],
            "title-idx": []
        }

        for idx, row in data_frame.iterrows():
            idn_i_list = row['predictions']
            title_i = row['title']
            for idn_i in idn_i_list:
                idn_i_str = gnd.pref_label_name(idn_i)
                if idn_i_str is None:
                    continue
                pair_dict["pair"].append((title_i, idn_i_str))
                pair_dict["label-ids"].append(idn_i)
                pair_dict["title-idx"].append(idx)

        ds = Dataset.from_dict(pair_dict)
        ds = ds.map(
            lambda x: self.tokenize(x["pair"]), batched=True, batch_size=2000)
        return ds
    
    def rerank(self, data_frame, gnd, bs=100):
        """
        Reranks the data frame using the reranker model.

        Args:
            data_frame (pd.DataFrame): The DataFrame containing the data. Needs columns 'predictions', 'title'.

        Returns:
            pd.DataFrame: The reranked DataFrame.
        """
        data_frame = data_frame.copy()
        ds = self.create_rerank_dataset(data_frame, gnd)
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label-ids', 'title-idx'])
        dataloader = DataLoader(ds, batch_size=bs, shuffle=False)
        sim = {
            "title-idx": [],
            "label-ids": [],
            "score": []
        }
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                scores = self.similarities(
                    batch["input_ids"],
                    batch["attention_mask"]
                )
                sim["title-idx"].extend(batch["title-idx"])
                sim["label-ids"].extend(batch["label-ids"])
                sim["score"].extend(scores.tolist())
        df_sim = pd.DataFrame(sim)
        df_sim["title-idx"] = df_sim["title-idx"].astype(int)

        index_unique = range(data_frame.shape[0])
        reranked_predictions = []
        scores = []
        for i in index_unique:
            df_i = df_sim[df_sim["title-idx"] == i]
            df_i = df_i.sort_values(by="score", ascending=False)
            reranked_predictions.append(df_i["label-ids"].tolist())
            scores.append(df_i["score"].tolist())
        
        data_frame["reranked-predictions"] = reranked_predictions
        data_frame["scores"] = scores

        return data_frame