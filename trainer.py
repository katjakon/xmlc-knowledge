from collections import OrderedDict
from logging import getLogger
import os
from statistics import mean

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import wandb

from utils import inference_tokenize, generate_predictions, BATCH_KEYS, SEP_TOKEN
from prompt_str import SUFFIX_PROMPT, PREFIX_PROMPT

logger = getLogger(__name__)

logging.set_verbosity_error()

class Trainer:

    def __init__(self, config) -> None:
        self.config = config
        self.model_name = config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.prompt_config = self.config["prompt_config"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.similarity_model = SentenceTransformer(self.config["sentence_transformer_model"])
        self.similarity_model.to(self.device)
    
    def save_checkpoint(self, model, scheduler, optimizer, output_dir):
        model_to_save = OrderedDict()
        for n, p in model.named_parameters():
            if p.requires_grad:
                model_to_save[n] = p
        if isinstance(model, torch.nn.DataParallel):
            model_to_save = {k.replace("module.", ""): v for k, v in model_to_save.items()}
        # Save the model
        if isinstance(model, torch.nn.DataParallel):
            model.module.save_pretrained(output_dir, state_dict=model_to_save)
        else:
            model.save_pretrained(output_dir, state_dict=model_to_save)
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    
    def eval_generate(self, model, eval_dataset):
        model.eval()
        model.pad_token_id = self.tokenizer.pad_token_id
        predictions = generate_predictions(
            model=model,
            tokenizer=self.tokenizer,
            dataset=eval_dataset,
            device=self.device
        )
        sims = []
        accs = []
        for i, pred_str in tqdm(enumerate(predictions), desc="Evaluating generation...", leave=False):
                labels = eval_dataset[i]["label-string"]
                label_list = eval_dataset[i]["label-names"]
                # Process generated text
                gen_embedding = self.similarity_model.encode(pred_str, convert_to_tensor=True)
                label_embedding = self.similarity_model.encode(labels, convert_to_tensor=True)
                similarity = self.similarity_model.similarity(gen_embedding, label_embedding)[0][0]
                sims.append(similarity.item())
                # Accuracy of predictions
                pred_label = set([word.strip() for word in pred_str.split(SEP_TOKEN)])
                gold_label = set(label_list)
                correct = len(pred_label.intersection(gold_label))
                total = len(gold_label)
                accuracy = correct / total if total > 0 else 0
                accs.append(accuracy)
        avg_similarity = mean(sims)
        avg_accuracy = mean(accs)
        model.train()
        return avg_similarity, avg_accuracy

    def evaluate(self, model, eval_dataloader):
        model.eval()
        eval_loss_list = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()} 
            with torch.no_grad():
                outputs = model(**batch)
                eval_loss = outputs.loss.mean()
                eval_loss_list.append(eval_loss.item())
        eval_loss = mean(eval_loss_list)
        model.train()
        return eval_loss
    
    def train(self, model, train_dataset, eval_dataset, output_dir):
        model.to(self.device)

        # Subsample for generation evaluation
        gen_len = min(200, len(eval_dataset))
        gen_eval_ds = eval_dataset.select(range(gen_len))
        gen_eval_ds = gen_eval_ds.map(lambda x: inference_tokenize(
            x, 
            self.tokenizer, 
            max_length=self.config["max_seq_length"], 
            suffix=SUFFIX_PROMPT, 
            prefix=PREFIX_PROMPT)
        )
        cols = [key for key in BATCH_KEYS if key in train_dataset.column_names]
        tensor_train_dataset = train_dataset.select_columns(cols).with_format("torch")
        tensor_eval_dataset = eval_dataset.select_columns(cols).with_format("torch")

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
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config["warmup_rate"]* total_steps),
            num_training_steps=total_steps,
        )

        global_step = 0
        track_loss = []
        best_eval_loss = float("inf")
        best_sim = 0.0
        for epoch in range(num_epochs):
            model.train()
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()} 
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                scheduler.step()

                track_loss.append(loss.item())
                if global_step % log_steps == 0:
                    # get current lr
                    lr = scheduler.get_lr()[0]
                    avg_loss = sum(track_loss) / len(track_loss)
                    track_loss = []
                    wandb.log({"loss": avg_loss, "learning_rate": lr, "Step": global_step})
                    print(f"\nStep {global_step}: Loss {avg_loss} Learning Rate {lr}")
                if global_step % eval_steps == 0:
                    eval_loss = self.evaluate(model, eval_dataloader)
                    wandb.log({"eval_loss": eval_loss})
                    print(f"Eval Loss: {eval_loss}")

                    if self.config["eval_generation"] is True:
                        gen_sim, gen_acc = self.eval_generate(model, gen_eval_ds)
                        wandb.log({"similarity": gen_sim})
                        wandb.log({"accuracy": gen_acc})
                        print(f"Generation Similarity: {gen_sim}")
                        print(f"Generation Accuracy: {gen_acc}")
                        if gen_sim > best_sim:
                            best_sim = gen_sim
                            check_path = os.path.join(output_dir, "best_model")
                            self.save_checkpoint(
                                model=model, 
                                scheduler=scheduler,
                                optimizer=optimizer,
                                output_dir=check_path)
                            print(f"Best model saved at step {global_step}")
                    else:
                        if eval_loss < best_eval_loss:
                            check_path = os.path.join(output_dir, "best_model")
                            self.save_checkpoint(
                                model=model, 
                                scheduler=scheduler,
                                optimizer=optimizer,
                                output_dir=check_path)
                            print(f"Best model saved at step {global_step}")
                if global_step % save_steps == 0:
                    check_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    self.save_checkpoint(
                                model=model, 
                                scheduler=scheduler,
                                optimizer=optimizer,
                                output_dir=check_path)
                global_step += 1
        wandb.finish()

