import argparse
import os
import pickle
import yaml

from datasets import Dataset
import pandas as pd
import torch
import transformers
from transformers import  pipeline, set_seed
from tqdm import tqdm

from data_collator import DataCollator
from gnd_graph import GNDGraph
from gnd_dataset import GNDDataset
from reranker import BGEReranker
from retriever import Retriever
from utils import load_model, generate_predictions, map_labels, process_output, SEP_TOKEN
from prompt_str import SYSTEM_PROMPT, USER_PROMPT, CONTEXT_PROMPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Predict labels for GND dataset using a model.")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--result_dir", type=str, help="Path to the result directory.")
parser.add_argument("--data_dir", type=str, help="Path to the data directory.")
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
data_dir = arguments.data_dir

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
gnd_graph = GNDGraph(gnd_graph)

# Load GND dataset
test_ds = pd.read_csv(
    os.path.join(data_dir, "sci-ger-ideal.tsv.gz"), 
    sep="\t", 
    compression="gzip",
    names=["title", "label_ids"])
test_ds = Dataset.from_pandas(test_ds) #.select(range(10))

retriever_model = config["sentence_transformer_model"]
retriever = Retriever(
    retriever_model=retriever_model,
    graph=gnd_graph,
    device=DEVICE,
)

retriever.load_search_index(
    index_path=index_path,
    mapping_path=mapping_path,
)

if do_hard_prompt:
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
            context_idns = retriever.retrieve_with_neighbors(
                texts= [title],
                top_k=config["context"]["top_k"],
                k=config["context"]["hops"],
                batch_size=1,
            )
            context_str = [
                [gnd_graph.pref_label_name(idn) for idn in idn_list]
                for idn_list in context_idns]
            keywords_str = f"{SEP_TOKEN} ".join(context_str[0])
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
    data_collator = DataCollator(
        tokenizer=tokenizer,
        graph=gnd_graph,  
        device=DEVICE,
        use_context=config["context"]["context_type"] is not None,
        top_k=config["context"]["top_k"],
    )
    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        collate_fn=lambda x: data_collator(x, inference=True),
    )

    raw_predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=loader,
        device=DEVICE,
        num_beams=1, 
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    processed_predictions = []
    for pred_str in raw_predictions:
        pred_str = pred_str.split(SEP_TOKEN)
        pred_str = [s.strip() for s in pred_str if s.strip()]  # Remove empty strings
        processed_predictions.append(pred_str)

mapped_predictions = map_labels(
    prediction_list=processed_predictions,
    retriever=retriever
)
reranker = BGEReranker("BAAI/bge-reranker-v2-m3", device=DEVICE)

pred_df = pd.DataFrame(
    {
        "predictions": mapped_predictions,
        "raw_predictions": raw_predictions,
        "title": test_ds["title"],
    }
)

pred_df = reranker.rerank(
    pred_df,
    gnd_graph,
    bs=200
)

# Reshape predictions necessary for QM.
index_file = os.path.join(data_dir, "sci-ger-ideal.arrow")
index_qm = pd.read_feather(index_file)

doc_ids = {i: index_qm[index_qm["location"] == i]["idn"].values[0] for i in tqdm(range(pred_df.shape[0]), desc="Map indices to doc idn")}

suggestions = []

for i, row in tqdm(pred_df.iterrows(), total=pred_df.shape[0], desc="Reshaping predictions..."):
    doc_id = doc_ids[i]
    reranked_predictions = row["reranked-predictions"]
    scores = row["scores"]
    for rank, (label_id, score) in enumerate(zip(reranked_predictions, scores)):
        suggestions.append({"doc_id": doc_id, "label_id": label_id, "score": score, "rank": rank + 1})
suggestions_df = pd.DataFrame(suggestions)  # Convert to dataframe
if do_hard_prompt:
    chp_str = "hard_prompt"
else:
    chp_str = f"checkpoint-{checkpoint}"
pred_df.to_csv(os.path.join(result_dir, f"predictions-{split}-{chp_str}-seed-{arguments.seed}.csv"))
suggestions_df.to_feather(os.path.join(result_dir, f"suggestions.arrow"))
