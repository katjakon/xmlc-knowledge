from adapters import AutoAdapterModel, BnConfig, init
from transformers import AutoTokenizer, AutoModelForCausalLM
from adapters.trainer import AdapterTrainer
from transformers.training_args import TrainingArguments 
from datasets import Dataset
import pickle
import evaluate
import torch
import wandb
import nltk
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.integrations import WandbCallback

from utils import PAD_TOKEN, SEP_TOKEN
from gnd_dataset import GNDDataset
from data_collator import DataCollator

KEYS = ["input_ids", "attention_mask", "labels", "seq_lengths"]

class CustomTrainer(AdapterTrainer):

    def evaluate_generation(
            self,
            eval_dataset = None,
            metric_key_prefix: str = "eval",
        ):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        self.model.eval()
        self.model.zero_grad()
        avg_meteor = 0.0
        avg_accuracy = 0.0
        count = 0
        num_instances = 300
        for instance in tqdm(eval_dataset, desc="Evaluating Generation", total=num_instances):
            count += 1
            if count > num_instances:
                break
            gold_labels = instance["label-names"]
            label_string = instance["label-string"]
            instance_batch = {k: torch.tensor(v).to(DEVICE) for k, v in instance.items() if k in KEYS}
            input_ids = instance_batch["input_ids"]
            attention_mask = instance_batch["attention_mask"]
            sequence_length = instance_batch["seq_lengths"]
            input_ids = input_ids[:sequence_length]
            attention_mask = attention_mask[:sequence_length]
            inputs = {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
            }
            output = self.model.generate(**inputs, pad_token_id=self.processing_class.eos_token_id)
            output_text = self.processing_class.decode(output[0], skip_special_tokens=True)
            output_labels = output_text.split(SEP_TOKEN)
            output_labels = [label.strip() for label in output_labels if label.strip()]
            if len(gold_labels) != 0:
                accuracy = len(set(gold_labels) & set(output_labels)) / len(set(gold_labels))
                avg_accuracy += accuracy
                results_meteor = meteor.compute(predictions=[output_text], references=[label_string])
                avg_meteor += results_meteor['meteor']
        return {f"{metric_key_prefix}_accuracy": avg_accuracy / len(eval_dataset),
                f"{metric_key_prefix}_meteor": avg_meteor / len(eval_dataset)}


    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        generation_metrics = self.evaluate_generation(eval_dataset, metric_key_prefix)
        self.log(generation_metrics)
        metrics.update(generation_metrics)
        return metrics



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

gnd = pickle.load(open("data/gnd.pickle", "rb"))
model_name = "meta-llama/Llama-3.2-1B"
ds_path = "datasets/no_context"


ds = GNDDataset(
    data_dir=ds_path,
    gnd_graph=gnd,
    load_from_disk=True

)

cm_model = AutoAdapterModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)

cm_model.add_causal_lm_head("lm_head", activation_function="silu")
bn_name = "bottleneck-adapter"
config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="silu")
cm_model.add_adapter(bn_name, config=config)
cm_model.set_active_adapters(bn_name)
cm_model.train_adapter(bn_name)

num_parameters = sum(p.numel() for p in cm_model.parameters() if p.requires_grad)
print(f"Number of trainable parameters in the adapter: {num_parameters}")

meteor = evaluate.load('meteor')
out_dir = "adapter_model"
training_args =  TrainingArguments(
    output_dir=out_dir,
    num_train_epochs=2,
    do_train=True,
    remove_unused_columns=False,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True
)
train_dataset = ds["train"]
eval_dataset = ds["validate"] # Use a small subset for evaluation

data_collator = DataCollator(
    tokenizer=tokenizer,
    graph=gnd,
    device=DEVICE,
)

trainer = CustomTrainer(
        model=cm_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=lambda x: data_collator(x, keys=["input_ids", "attention_mask", "labels"]),
    )

trainer.train() 
trainer.save_model(out_dir)
