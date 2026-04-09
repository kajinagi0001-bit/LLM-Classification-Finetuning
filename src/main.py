import argparse
from train import train
from inference import inference
from config import CFG

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "infer", "test"])
parser.add_argument("-w", "--wandb", action='store_true')

args = parser.parse_args()
CFG.use_wandb = args.wandb

if args.mode == "train":
    best_epoch, best_val_loss = train(CFG)
elif args.mode == "infer":
    inference()
elif args.mode == "test":
    print("test mode")