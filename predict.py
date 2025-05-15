import argparse
import os
import pickle
import re
import yaml

import pandas as pd
import torch
import transformers
from transformers import  pipeline
from tqdm import tqdm

from gnd_dataset import GNDDataset
from retriever import Retriever
from utils import load_model, generate_predictions, get_label_mapping, map_labels, process_output, SEP_TOKEN
from prompt_str import SYSTEM_PROMPT, USER_PROMPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--data_dir", type=str, help="Path to the GND dataset directory.")
parser.add_argument("--gnd_graph", type=str, help="Path to the GND graph file (pickle).")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--result_dir", type=str, help="Path to the result directory.")
parser.add_argument("--hard-prompt", action="store_true", help="Only hard prompt the model.", default=False)
parser.add_argument("--hard_prompt_model", help="Name of the model to use.", default=None)
parser.add_argument("--index", type=str, help="Path to the index file.")
parser.add_argument("--mapping", type=str, help="Path to the label mapping file.")

arguments = parser.parse_args()
data_dir = arguments.data_dir
gnd_graph = arguments.gnd_graph
config_path = arguments.config
result_dir = arguments.result_dir
do_hard_prompt = arguments.hard_prompt
hard_prompt_model = arguments.hard_prompt_model
index_path = arguments.index
mapping_path = arguments.mapping

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

exp_name = config["experiment_name"]
model_name = config["model_name"]
if do_hard_prompt:
    if hard_prompt_model is not None:
        model_name = hard_prompt_model
    model_str = model_name.split("/")[-1]
    exp_name = f"{model_str}-hard-prompt"
    
result_dir = os.path.join(result_dir, exp_name)

if os.path.exists(result_dir):
    print(f"Result directory {result_dir} already exists. Please remove it or choose a different name.")
    exit(1)
os.makedirs(result_dir)

# Load GND graph
gnd_graph = pickle.load(open(gnd_graph, "rb"))

# Load GND dataset
gnd_ds = GNDDataset(
    data_dir=data_dir,
    gnd_graph=gnd_graph,
    config=config
)

if do_hard_prompt:
    test_ds = gnd_ds["test"]
    raw_predictions = []
    count = 0
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    for row in tqdm(test_ds):
        title = row["title"]
        messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(title)},
        ]
        outputs = pipe(messages, num_return_sequences=1, do_sample=True, temperature=0.7)
        new_tokens = outputs[0]["generated_text"][-1]["content"]
        raw_predictions.append(new_tokens.strip())
    
    processed_predictions = []
    for pred_str in raw_predictions:
        pred_str = process_output(pred_str)
        processed_predictions.append(pred_str)

else:
    output_dir = config["checkpoint_path"]
    output_dir = os.path.join(output_dir, exp_name)
    checkpoint_path = os.path.join(output_dir, "best_model", "model.safetensors")

    model, tokenizer = load_model(checkpoint_path, config=config, device=DEVICE, data_parallel=True)

    gnd_ds.inference_tokenize_datasets(tokenizer=tokenizer, splits=["test"])
    test_ds = gnd_ds["test"]

    raw_predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=test_ds,
        device=DEVICE
    )

    processed_predictions = []
    for pred_str in raw_predictions:
        pred_str = pred_str.split(SEP_TOKEN)
        pred_str = [s.strip() for s in pred_str if s.strip()]  # Remove empty strings
        processed_predictions.append(pred_str)


retriever_model = config["sentence_transformer_model"]
retriever = Retriever(
    retriever_model=retriever_model,
    device=DEVICE,
)

index = pickle.load(open(index_path, "rb"))
mapping = pickle.load(open(mapping_path, "rb")) 

mapped_predictions = map_labels(
    prediction_list=processed_predictions,
    index=index,
    retriever=retriever,
    label_mapping=mapping
)


pred_df = pd.DataFrame(
    {
        "predictions": mapped_predictions,
        "raw_predictions": raw_predictions,
        "label-ids": test_ds["label-ids"],
        "label-names": test_ds["label-names"],
        "title": test_ds["title"],
    }
)

pred_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)