from collections import OrderedDict
from logging import getLogger
import os
import pickle

from datasets import Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from safetensors import safe_open
import wandb
import yaml

from utils import init_prompt_model, get_pref_label, strip_uri, tokenize
from prompt_str import SUFFIX_PROMPT, PREFIX_PROMPT

logger = getLogger(__name__)


class Trainer:

    def __init__(self, config) -> None:
        self.config = config
        self.model_name = config["model_name"]
        self.prompt_config = config["prompt_config"]
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"

    def load_checkpoint(self, checkpoint_path, config, device):
        prompt_config = config["prompt_config"]
        model_name = config["model_name"]
        model, tokenizer = init_prompt_model(model_name, prompt_config)
        tensors = {}
        with safe_open(checkpoint_path, framework="pt", device=device) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
        return model, tokenizer
    
    def save_checkpoint(self, model, output_dir):
        model_to_save = OrderedDict()
        for n, p in model.named_parameters():
            if p.requires_grad:
                model_to_save[n] = p
        model.save_pretrained(output_dir, state_dict=model_to_save)
    
    def evaluate(self, model, eval_dataloader):
        model.eval()
        eval_loss = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
        eval_loss /= len(eval_dataloader)
        wandb.log({"eval_loss": eval_loss})
        print(f"\nEval Loss: {eval_loss}")
        model.train()
        return eval_loss
    
    def train(self, model, train_dataset, eval_dataset, output_dir):
        model.to(self.device)
        tensor_train_dataset = train_dataset.select_columns(["input_ids", "attention_mask", "labels"]).with_format("torch")
        tensor_eval_dataset = eval_dataset.select_columns(["input_ids", "attention_mask", "labels"]).with_format("torch")

        # Create DataLoader
        train_dataloader = DataLoader(tensor_train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        eval_dataloader = DataLoader(tensor_eval_dataset, batch_size=self.config["batch_size"], shuffle=False)


        num_epochs = self.config["num_epochs"]
        log_steps = self.config["logging_steps"]
        eval_steps = self.config["eval_steps"]
        save_steps = self.config["save_steps"]
        max_norm = self.config["max_grad_norm"]
        lr = float(self.config["learning_rate"])
        total_steps = len(train_dataloader) * num_epochs

        # Create optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0 - (step / total_steps))

        global_step = 0
        for epoch in range(num_epochs):
            model.train()
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                scheduler.step()
                if global_step % log_steps == 0:
                    # get current lr
                    lr = scheduler.get_last_lr()[0]
                    wandb.log({"loss": loss.item(), "learning_rate": lr})
                    print(f"\nStep {global_step}: Loss {loss.item()} Learning Rate {lr}")
                if global_step % eval_steps == 0:
                    self.evaluate(model, eval_dataloader)
                global_step += 1
                if global_step % save_steps == 0:
                    self.save_checkpoint(model, os.path.join(output_dir, f"checkpoint-{global_step}"))
        wandb.finish()



if __name__ == "__main__":
    def apply_pref_label(label_list, graph):
        return [get_pref_label(graph, label) for label in label_list if label in graph.nodes]

    data_path = "data/title/train.tsv.gz"
    eval_path = "data/title/validate.tsv.gz"
    data_df = pd.read_csv(data_path, sep="\t", compression="gzip", header=0, names=["title", "label-idn"])
    data_df["label-idn"] = data_df["label-idn"].apply(strip_uri)
    eval_df = pd.read_csv(eval_path, sep="\t", compression="gzip", header=0, names=["title", "label-idn"])
    eval_df["label-idn"] = eval_df["label-idn"].apply(strip_uri)

    gnd = pickle.load(open("data/gnd.pickle", "rb"))
    data_df["label_list"] = data_df["label-idn"].apply(lambda x: apply_pref_label(x, gnd))
    eval_df["label_list"] = eval_df["label-idn"].apply(lambda x: apply_pref_label(x, gnd))
    print(data_df.head())

    # Subsample the dataset
    number = 150_000
    eval_number = 1500
    data_df = data_df.sample(number, random_state=42)
    eval_df = eval_df.sample(eval_number, random_state=42)

    train_ds = Dataset.from_pandas(data_df)
    eval_ds = Dataset.from_pandas(eval_df)

    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load the model and tokenizer
    model_name = config["model_name"]
    prompt_config = config["prompt_config"]
    max_seq_length = config["max_seq_length"]
    model, tokenizer = init_prompt_model(model_name, prompt_config=prompt_config)
    train_ds = train_ds.map(lambda x: tokenize(x, tokenizer, max_length=max_seq_length, suffix=SUFFIX_PROMPT, prefix=PREFIX_PROMPT))
    eval_ds = eval_ds.map(lambda x: tokenize(x, tokenizer, max_length=max_seq_length, suffix=SUFFIX_PROMPT, prefix=PREFIX_PROMPT))

    wandb.init(
      # Set the project where this run will be logged
      project="xmlc-knowledge",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"baseline-{model_name}",
      # Track hyperparameters and run metadata
      config={
          "model_name": model_name,
          "config": config,
          "n_train": number,
      })

    # # Initialize the trainer
    trainer = Trainer(config)
    output_dir = "checkpoints"
    if output_dir not in os.listdir():
        os.mkdir(output_dir)
    trainer.train(model, train_ds, eval_ds, output_dir=output_dir)

