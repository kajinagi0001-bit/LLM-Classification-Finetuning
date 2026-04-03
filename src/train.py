# src/train.py
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from config import CFG
from dataset import preprocess, LMSYSDataset
from model import PairwiseDebertaClassifier
import json
import wandb

wandb.init(project="LLM-Classification Finetuning", name=CFG.exp_name, config=CFG)

def train():
    exp_dir = f"output/exp/{CFG.exp_name}"
    os.makedirs(exp_dir, exist_ok=True)

    cfg_dict = {
        k: v for k, v in CFG.__dict__.items()
        if not k.startswith("__") and not callable(v)
    }

    with open(f"{exp_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=4, ensure_ascii=False)


    os.makedirs("output/models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("input/train.csv")
    df = preprocess(df)

    df["class_name"] = df[["winner_model_a", "winner_model_b", "winner_tie"]].idxmax(axis=1)
    df["label"] = df["class_name"].map(CFG.name2label)

    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=CFG.seed,
        stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    train_ds = LMSYSDataset(train_df, tokenizer, CFG.max_length)
    valid_ds = LMSYSDataset(valid_df, tokenizer, CFG.max_length)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False)

    model = PairwiseDebertaClassifier(CFG.model_name, CFG.num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = AdamW(model.parameters(), lr=CFG.lr)

    num_training_steps = len(train_loader) * CFG.epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    def run_epoch(loader, model, criterion, optimizer=None, scheduler=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()
        total_loss = 0
        total_correct = 0
        total_count = 0
        
        for batch in tqdm(loader):
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            labels = batch["label"].to(device)
            
            with torch.set_grad_enabled(is_train):
                logits = model(
                    input_ids_a = input_ids_a,
                    attention_mask_a = attention_mask_a,
                    input_ids_b = input_ids_b,
                    attention_mask_b = attention_mask_b
                )
                loss = criterion(logits, labels)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

        return total_loss / total_count, total_correct / total_count

    best_val_loss = float("inf")

    for epoch in range(CFG.epochs):
        train_loss, train_acc = run_epoch(train_loader, model, criterion, optimizer, scheduler)
        val_loss, val_acc = run_epoch(valid_loader, model, criterion)

        print(f"Epoch {epoch+1}")
        print(f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
        print(f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"output/exp/{CFG.exp_name}/best_model.pth")

if __name__ == "__main__":
    train()