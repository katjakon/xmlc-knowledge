import argparse
import os
import pickle
import re
import yaml

import pandas as pd
import torch
import transformers
from transformers import  pipeline, set_seed
from tqdm import tqdm

from gnd_dataset import GNDDataset
from retriever import Retriever
from utils import load_model, generate_predictions, map_labels, process_output, SEP_TOKEN
from prompt_str import SYSTEM_PROMPT, USER_PROMPT, CONTEXT_PROMPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--result_dir", type=str, help="Path to the result directory.")
parser.add_argument("--hard-prompt", action="store_true", help="Only hard prompt the model.", default=False)
parser.add_argument("--index", type=str, help="Path to the index file.")
parser.add_argument("--mapping", type=str, help="Path to the mapping file.")
parser.add_argument("--split", type=str, help="Split to use for evaluation.", default="test")
parser.add_argument("--checkpoint", type=str, help="Checkpoint to use for evaluation.", default="best_model")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

arguments = parser.parse_args()
config_path = arguments.config
result_dir = arguments.result_dir
do_hard_prompt = arguments.hard_prompt
index_path = arguments.index
mapping_path = arguments.mapping
split = arguments.split
checkpoint = arguments.checkpoint

# Set random seed for reproducibility
set_seed(arguments.seed)

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

exp_name = config["experiment_name"]
model_name = config["model_name"]
    
result_dir = os.path.join(result_dir, exp_name)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Load GND graph
gnd_path  = config["graph_path"]
gnd_graph = pickle.load(open(gnd_path, "rb"))


data_dir = config["dataset_path"]
# Load GND dataset
gnd_ds = GNDDataset(
    data_dir=data_dir,
    gnd_graph=gnd_graph,
    config=config,
    load_from_disk=True,
)

if do_hard_prompt:
    test_ds = gnd_ds[split]
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

        if config["context"]["context_type"] is not None:
            keywords = row["context_str"]
            keywords = [kw.strip() for kw in keywords if kw.strip()]
            keywords_str = SEP_TOKEN.join(keywords)
            system_prompt = SYSTEM_PROMPT + CONTEXT_PROMPT.format(keywords_str)
        else:
            system_prompt = SYSTEM_PROMPT
        messages = [
        {"role": "system", "content": system_prompt},
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
    checkpoint_path = os.path.join(output_dir, checkpoint, "model.safetensors")

    model, tokenizer = load_model(checkpoint_path, config=config, device=DEVICE, data_parallel=True)

    gnd_ds.inference_tokenize_datasets(tokenizer=tokenizer, splits=[split], with_context=config["context"]["in_prompt"], max_context=config["context"]["n_context"])
    test_ds = gnd_ds[split]

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
if do_hard_prompt:
    chp_str = "hard_prompt"
else:
    chp_str = f"checkpoint-{checkpoint}"
pred_df.to_csv(os.path.join(result_dir, f"predictions-{split}-{chp_str}-seed-{arguments.seed}.csv"))