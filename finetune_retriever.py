from sentence_transformers import SentenceTransformer,  losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset
import pickle
from tqdm import tqdm
import yaml

from gnd_dataset import GNDDataset

gnd_path = "data/gnd.pickle"
config_path = "configs/config_pt_baseline.yaml"
# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
gnd_graph = pickle.load(open(gnd_path, "rb"))

gnd_ds = GNDDataset(
    data_dir=config["dataset_path"],
    gnd_graph=gnd_graph,
    config=config,
    load_from_disk=True,
)
subset = range(100_000)
train_ds = gnd_ds["train"].select(subset)

model = SentenceTransformer("BAAI/bge-m3")

retriever_dict = {
    "anchor": [],
    "positive": [],
}
for i in tqdm(train_ds):
    gold_labels_ids = i["label-ids"]
    gold_labels = i["label-names"]
    title = i["title"]
    for keyword in gold_labels:
        retriever_dict["anchor"].append(title)
        retriever_dict["positive"].append(keyword)

eval_ds = gnd_ds["validate"].select(range(1000))
eval_dict = {
    "anchor": [],
    "positive": [],
}
for i in tqdm(eval_ds):
    gold_labels_ids = i["label-ids"]
    gold_labels = i["label-names"]
    title = i["title"]
    for keyword in gold_labels:
        eval_dict["anchor"].append(title)
        eval_dict["positive"].append(keyword)

train_dataset = Dataset.from_dict(retriever_dict)
eval_dataset = Dataset.from_dict(eval_dict) 

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="retriever/partial",
    # Optional training parameters:
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    eval_strategy="steps",
    eval_steps=1000,
    # Optional tracking/debugging parameters:
    save_total_limit=2,
    logging_steps=500,
    run_name="partial",  # Will be used in W&B if `wandb` is installed
)

loss = losses.MultipleNegativesRankingLoss(model)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    eval_dataset=eval_dataset,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()

