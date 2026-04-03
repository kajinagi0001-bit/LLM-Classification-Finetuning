# src/inference.py
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from config import CFG
from dataset import preprocess, make_pairs, LMSYSDataset
from model import PairwiseDebertaClassifier


def inference():
    os.makedirs(f"output/exp/{CFG.exp_name}", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv("input/test.csv")
    test_df = preprocess(test_df)
    test_df = test_df.apply(make_pairs, axis=1)
    test_df.encode_fail.value_counts(normalize=False)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    test_ds = LMSYSDataset(test_df, tokenizer, CFG.max_length, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False)

    model = PairwiseDebertaClassifier(CFG.model_name, CFG.num_classes).to(device)
    model.load_state_dict(torch.load(f"output/exp/{CFG.exp_name}/best_model.pth", map_location=device))
    model.eval()

    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs).numpy()

    submission = test_df[["id"]].copy()
    submission["winner_model_a"] = all_probs[:, 0]
    submission["winner_model_b"] = all_probs[:, 1]
    submission["winner_tie"] = all_probs[:, 2]

    submission.to_csv(f"output/exp/{CFG.exp_name}/submission.csv", index=False)
    print(f"saved to output/exp/{CFG.exp_name}/submission.csv")

if __name__ == "__main__":
    inference()