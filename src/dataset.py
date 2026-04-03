# src/dataset.py
import json
import pandas as pd
import torch
from torch.utils.data import Dataset

def preprocess(df):
    df["prompt"] = df["prompt"].map(lambda x: json.loads(x)[0] if pd.notna(x) else "")
    df["response_a"] = df["response_a"].map(lambda x: json.loads(x)[0] if pd.notna(x) else "")
    df["response_b"] = df["response_b"].map(lambda x: json.loads(x)[0] if pd.notna(x) else "")

    def build_text(prompt, response):
        return f"Prompt: {prompt}\n\nResponse: {response}"

    df["text_a"] = df.apply(lambda r: build_text(r["prompt"], r["response_a"]), axis=1)
    df["text_b"] = df.apply(lambda r: build_text(r["prompt"], r["response_b"]), axis=1)
    return df


class LMSYSDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        enc_a = self.tokenizer(
            row["text_a"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        enc_b = self.tokenizer(
            row["text_b"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        item = {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0)
        }

        if not self.is_test:
            item["label"] = torch.tensor(row["label"], dtype=torch.long)

        return item