class CFG:
    seed = 42
    model_name = "microsoft/deberta-v3-small"
    exp_name = "exp007_lr1e6" # 実験ごとに変更
    max_length = 512
    batch_size = 8
    epochs = 10
    lr = 1e-6
    num_classes = 3
    label2name = {0: 'winner_model_a', 1: 'winner_model_b', 2: 'winner_tie'}
    name2label = {v: k for k, v in label2name.items()}
    mini_data = True
    use_wandb = False
    amp = True