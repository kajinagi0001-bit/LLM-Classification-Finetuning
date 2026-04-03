import argparse
from train import train
from inference import inference

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "infer", "test"])

args = parser.parse_args()

if args.mode == "train":
    train()
elif args.mode == "infer":
    inference()
elif args.mode == "test":
    print("test mode")