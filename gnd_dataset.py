import os
from typing import List, Dict, Any

from datasets import Dataset
import pandas as pd

from prompt_str import PREFIX_PROMPT, SUFFIX_PROMPT
from utils import tokenize, inference_tokenize, get_pref_label, strip_uri


class GNDDataset:
    """
    A custom dataset class for the GND (German National Library) dataset.
    Inherits from the Hugging Face `Dataset` class.
    """

    FILES = {
        "train": "train.tsv.gz",
        "validate": "validate.tsv.gz",
        "test": "test.tsv.gz",
    }
    def __init__(self, data_dir: str, gnd_graph: str, config: Dict[str, Any], load_from_disk: bool = False):
        """
        Initializes the GNDDataset.

        Args:
            data_dir (str): The directory where the GND dataset is stored.
            gnd_graph (networkx): GND graph file (network x object.).

        """
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.sort_by_freq = bool(self.config["sort_by_freq"]) # Sort labels by frequency
        self.use_k_freq_labels = int(self.config["use_k_freq_labels"]) # Only use k most frequent labels
        self.gnd_graph = gnd_graph
        if load_from_disk:
            self.dataset = self.load_from_disk(self.data_dir)
        else:
            self.dataset = self.load_dataset()
    
    def label_frequency(self, labels: List[List[str]]) -> Dict[str, int]:
        """
        Computes the frequency of each label in the dataset.

        Args:
            labels (List[List[str]]): List of lists containing labels.

        Returns:
            Dict[str, int]: A dictionary with labels as keys and their frequencies as values.
        """
        label_freq = {}
        for label_list in labels:
            for label in label_list:
                if label not in label_freq:
                    label_freq[label] = 0
                label_freq[label] += 1
        return label_freq
    
    def labels_string(self, labels: List[List[str]]) -> List[str]:
        """
        Converts a list of lists of labels into a list of strings.

        Args:
            labels (List[List[str]]): List of lists containing labels.

        Returns:
            List[str]: A list of strings representing the labels.
        """
        string_list = []
        for label_list in labels:
            l_str = ""
            n_labels = len(label_list)
            for i, label in enumerate(label_list):
                l_str += label 
                if i < n_labels - 1:
                    l_str += ", "
            string_list.append(l_str)
        return string_list


    def load_dataset(self) -> Dict[str, Dataset]:
        """
        Loads the GND dataset from the specified directory.

        Returns:
            Dataset: The loaded GND dataset.
        """

        for split, split_file in self.FILES.items():
            file_path = os.path.join(self.data_dir, split_file)
            df = pd.read_csv(file_path, sep="\t", compression="gzip", header=0, names=["title", "label-ids"])
            # Subsample the dataset
            if split == "train":
                # Subsample the training set
                df = df.sample(frac=self.config["train_subsample_ratio"], random_state=42)
                train_df = df
            elif split == "validate":
                # Subsample the validation set
                df = df.sample(frac=self.config["validate_subsample_ratio"], random_state=42)
                validate_df = df
            elif split == "test":
                test_df = df

        # Get prefered label namens:
        for split_df in [train_df, validate_df, test_df]:
            split_df["label-ids"] = split_df["label-ids"].apply(strip_uri)
            split_df["label-names"] = split_df["label-ids"].apply(
                lambda idns: [get_pref_label(self.gnd_graph, label) for label in idns if label in self.gnd_graph.nodes]
            )
            # Get label frequency
            if self.sort_by_freq:
                label_freq = self.label_frequency(split_df["label-names"].tolist())
                # Sort labels by frequency
                split_df["label-names"] = split_df["label-names"].apply(
                    lambda x: sorted(x, key=lambda y: label_freq.get(y, 0), reverse=True)
                    )
            # Limit to k most frequent labels
            if self.use_k_freq_labels > 0:
                split_df["label-names"] = split_df["label-names"].apply(
                    lambda x: x[:self.use_k_freq_labels]
                )
            
            # Convert label-names to string
            split_df["label-string"] = self.labels_string(split_df["label-names"].tolist())
        # Convert DataFrames to Datasets
        train_dataset = Dataset.from_pandas(train_df)
        validate_dataset = Dataset.from_pandas(validate_df)
        test_dataset = Dataset.from_pandas(test_df)
        return {
            "train": train_dataset,
            "validate": validate_dataset,
            "test": test_dataset,
        }
    
    def tokenize_datasets(self, tokenizer, splits=None):
        """
        Tokenizes the datasets using the provided tokenizer.

        Args:
            tokenizer: The tokenizer to use for tokenization.
            splits (List[str], optional): List of dataset splits to tokenize. If None, all splits are tokenized.

        Returns:
            None
        """
        for split, dataset in self.dataset.items():
            if splits is not None and split not in splits:
                continue
            dataset = dataset.map(
                lambda x: tokenize
                (
                    x, 
                    tokenizer, 
                    max_length=self.config["max_seq_length"], 
                    suffix=SUFFIX_PROMPT, 
                    prefix=PREFIX_PROMPT
                ))
            self.dataset[split] = dataset
    
    def inference_tokenize_datasets(self, tokenizer, splits=None):
        """
        Tokenizes the datasets using the provided tokenizer for inference.

        Args:
            tokenizer: The tokenizer to use for tokenization.
            splits (List[str], optional): List of dataset splits to tokenize. If None, all splits are tokenized.

        Returns:
            None
        """
        for split, dataset in self.dataset.items():
            if splits is not None and split not in splits:
                continue
            dataset = dataset.map(
                lambda x: inference_tokenize
                (
                    x, 
                    tokenizer, 
                    max_length=self.config["max_seq_length"], 
                    suffix=SUFFIX_PROMPT, 
                    prefix=PREFIX_PROMPT
                ))
            self.dataset[split] = dataset
    
    def save_to_disk(self, path):
        """
        Saves the dataset to disk.

        Args:
            path (str): The path to save the dataset.
        """
        for split, dataset in self.dataset.items():
            dataset.save_to_disk(os.path.join(path, split))
    
    def load_from_disk(self, path):
        """
        Loads the dataset from disk.

        Args:
            path (str): The path to load the dataset from.
        """
        ds = {}
        for key in self.FILES.keys():
            split_path = os.path.join(path, key)
            ds[key] = Dataset.load_from_disk(split_path)
        return ds

    
    def __getitem__(self, split):
        return self.dataset[split]

    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        return repr(self.dataset)


            
        