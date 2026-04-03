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

wandb.init(project="LLM-Classification Finetuning", name=CFG.exp_name, config=CFG.__dict__)

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

    steps = len(train_loader) * CFG.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(0.1 * steps),
        steps
    )

    def run_epoch(loader, train=False):
        model.train() if train else model.eval()
        total_loss = 0

        for batch in tqdm(loader):
            labels = batch["label"].to(device)

            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}

            with torch.set_grad_enable(train):
                logits = model(**inputs)
                loss = criterion(logits, labels)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    best_val_loss = float("inf")

    for epoch in range(CFG.epochs):
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(valid_loader)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        print(epoch + 1, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"output/exp/{CFG.exp_name}/best_model.pth")

if __name__ == "__main__":
    train()