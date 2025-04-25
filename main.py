import argparse
import os
import pickle
import re
import yaml

import pandas as pd
import torch
import wandb

from gnd_dataset import GNDDataset
from trainer import Trainer
from retriever import Retriever
from utils import init_prompt_model, generate_predictions, get_label_mapping, map_labels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--data_dir", type=str, help="Path to the GND dataset directory.")
parser.add_argument("--gnd_graph", type=str, help="Path to the GND graph file (pickle).")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--result_dir", type=str, help="Path to the result directory.")

arguments = parser.parse_args()
data_dir = arguments.data_dir
gnd_graph = arguments.gnd_graph
config_path = arguments.config
result_dir = arguments.result_dir

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

exp_name = config["experiment_name"]

output_dir = config["checkpoint_path"]
output_dir = os.path.join(output_dir, exp_name)

if os.path.exists(output_dir):
    print(f"Output directory {output_dir} already exists. Please remove it or choose a different name.")
    exit(1)

os.makedirs(output_dir)

res_dir = os.path.join(result_dir, exp_name)
if os.path.exists(res_dir):
    print(f"Result directory {res_dir} already exists. Please remove it or choose a different name.")
    exit(1)

os.makedirs(res_dir)

# Load GND graph
gnd_graph = pickle.load(open(gnd_graph, "rb"))

model_name = config["model_name"]

model, tokenizer = init_prompt_model(
    model_name=model_name,
    prompt_config=config["prompt_config"]
)

model = torch.nn.DataParallel(model)

# Load GND dataset
gnd_ds = GNDDataset(
    data_dir=data_dir,
    gnd_graph=gnd_graph,
    config=config
)

# Tokenize the datasets
gnd_ds.tokenize_datasets(tokenizer=tokenizer, splits=["train", "validate"])
gnd_ds.inference_tokenize_datasets(tokenizer=tokenizer, splits=["test"])

# Split the dataset into train, validation, and test sets
train_ds = gnd_ds["train"]
valid_ds = gnd_ds["validate"]
test_ds = gnd_ds["test"]

# How many parameters are in the model?
total_model_params = 0
num_trained_params = 0
for p in model.parameters():
    if p.requires_grad:
        num_trained_params += p.numel()
    else:
        total_model_params += p.numel()
print("Total Model Parameters: {}, Trainable Parameters: {}, Percentage {}".format(
    total_model_params, 
    num_trained_params,
    num_trained_params / (total_model_params + num_trained_params)
    )
    )
prompt_type = config["prompt_config"]["type"]
layer = config["prompt_config"]["at_layer"]
num_tokens = config["prompt_config"]["num_prompt_tokens"]

wandb.init(
      # Set the project where this run will be logged
      project="xmlc-knowledge",
      name=f"{exp_name}",
      # Track hyperparameters and run metadata
      config={
          "model_name": model_name,
          "config": config,
          "n_train": len(train_ds),
          "params": num_trained_params,
      })

trainer = Trainer(config)
trainer.train(
    model=model,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    output_dir=output_dir,
)

# test_ds = test_ds.select(range(100))  # Select a subset of the test dataset for evaluation

raw_predictions = generate_predictions(
    model=model,
    tokenizer=tokenizer,
    dataset=test_ds,
    device=DEVICE
)

processed_predictions = []
for pred_str in raw_predictions:
    pred_str = re.split(r"[,;]", pred_str)  # Split by commas or semicolons
    pred_str = [s.strip() for s in pred_str if s.strip()]  # Remove empty strings
    processed_predictions.append(pred_str)


# Map raw labels to GND labels
label_strings, label_mapping = get_label_mapping(gnd_graph)

retriever_model = config["sentence_transformer_model"]
retriever = Retriever(
    retriever_model=retriever_model,
    device=DEVICE,
)

index = retriever.fit(labels=label_strings, batch_size=512)

mapped_predictions = map_labels(
    prediction_list=processed_predictions,
    index=index,
    retriever=retriever,
    label_mapping=label_mapping,
    label_strings=label_strings
)


pred_df = pd.DataFrame(
    {
        "predictions": mapped_predictions,
        "raw_predictions": raw_predictions,
        "label-ids": test_ds["label-idns"],
        "label-names": test_ds["label-names"],
        "title": test_ds["title"],
    }
)

pred_df.to_csv(os.path.join(res_dir, "predictions.csv"), index=False)