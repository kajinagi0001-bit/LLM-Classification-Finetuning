# src/train.py
import os
import pandas as pd
from model import MODEL_DICT
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from config import CFG

from dataset import preprocess, make_pairs, LMSYSDataset
from model.model import PairwiseDebertaClassifier, DebertaClassifier_TokenAttentionPooling
import json
import wandb
from torch.amp import autocast, GradScaler

#scaler = GradScaler()

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
    if CFG.mini_data:
        df = df.iloc[:len(df) // 3]
    df = preprocess(df)
    df = df.apply(make_pairs, axis=1)
    df.encode_fail.value_counts(normalize=False)

    df["class_name"] = df[["winner_model_a", "winner_model_b", "winner_tie"]].idxmax(axis=1)
    df["label"] = df["class_name"].map(CFG.name2label)

    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=CFG.seed,
        stratify=df["label"]
    )

    #print(train_df.iloc[0]["text_a"][:1000])
    #print("=" * 80)
    #print(train_df.iloc[0]["text_b"][:1000])
    #print(train_df.iloc[0]["label"])
    #exit()

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_encoder_name)

    #sample = train_df.iloc[0]
    #tokens_a = tokenizer(sample["text_a"], truncation=False)["input_ids"]
    #tokens_b = tokenizer(sample["text_b"], truncation=False)["input_ids"]
    #print("len_a:", len(tokens_a))
    #print("len_b:", len(tokens_b))


    train_ds = LMSYSDataset(train_df, tokenizer, CFG.max_length)
    valid_ds = LMSYSDataset(valid_df, tokenizer, CFG.max_length)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False)

    model = MODEL_DICT[CFG.model_structure_name](CFG.model_encoder_name, CFG.num_classes).to(device).float()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=0.01)
    num_training_steps = len(train_loader) * CFG.epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    wandb.watch(model, log='all')

    def run_epoch(loader, model, criterion, optimizer=None, scheduler=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()
        total_loss = 0
        total_correct = 0
        total_count = 0
        
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            labels = batch["label"].to(device)
            #print(f"input_ids_a: {input_ids_a.shape}, {input_ids_a}\n")
            #print(f"attention_mask_a: {attention_mask_a.shape}, {attention_mask_a}\n")
            #print(f"input_ids_b: {input_ids_b.shape}, {input_ids_b}\n")
            #print(f"attention_mask_b: {attention_mask_b.shape}, {attention_mask_b}\n")
            #print(labels)

            #before = model.classifier.weight.detach().clone()
            with torch.set_grad_enabled(is_train):
                #with autocast('cuda', enabled=CFG.amp):
                logits = model(
                    input_ids_a = input_ids_a,
                    attention_mask_a = attention_mask_a,
                    input_ids_b = input_ids_b,
                    attention_mask_b = attention_mask_b
                )

                #print(f"logits[0]: {logits}")
                #print(f"labels[0]: {labels}")
                loss = criterion(logits, labels)
                #print(f"loss[0]: {loss}")

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    #scaler.scale(loss).backward()
                    #scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    #scaler.step(optimizer)
                    scheduler.step()
                    #scaler.update()
            
            #total_norm = 0.0
            #for name, p in model.named_parameters():
            #    if p.grad is not None:
            #        param_norm = p.grad.data.norm(2).item()
            #        total_norm += param_norm ** 2
            #        if "classifier" in name:
            #            print(name, param_norm)
            #total_norm = total_norm ** 0.5
            #print("grad_norm:", total_norm)
            #fter = model.classifier.weight.detach().clone()
            #print("update size:", (after - before).abs().mean().item())

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)
            #debug
            #print(f"batch_{i} total_loss: {total_loss}, total_correct: {total_correct}, total_count: {total_count}")

        return total_loss / total_count, total_correct / total_count

    best_val_loss = float("inf")

    for epoch in range(CFG.epochs):
        train_loss, train_acc = run_epoch(train_loader, model, criterion, optimizer, scheduler)
        val_loss, val_acc = run_epoch(valid_loader, model, criterion)

        print(f"Epoch {epoch+1}")
        print(f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
        print(f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"output/exp/{CFG.exp_name}/best_model.pth")

if __name__ == "__main__":
    train()