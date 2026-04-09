# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel


# DeBERTa classifier model
class PairwiseDebertaClassifier(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size*2, num_classes)

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        return summed / denom
    
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        #print(f"outputs: {outputs[0]}, size: {outputs.shape}")
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        #print(f"pooled: {pooled[0]}, size: {pooled.shape}")
        return pooled
        #return outputs.last_hidden_state[:, 0].float()
    
    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        feat_a = self.encode(input_ids_a, attention_mask_a)
        #print(f"feat_a: {feat_a[0]}")
        feat_b = self.encode(input_ids_b, attention_mask_b)
        #print(f"feat_b: {feat_b[0]}")
        feats = torch.cat([feat_a, feat_b], dim=1)
        feats = feats
        #print(f"feats: {feats[0]}")
        logits = self.classifier(feats)
        #print(f"logits: {logits[0]}")
        return logits.float()


# DeBERTa classifier model
class DebertaClassifier_TokenAttentionPooling(nn.Module):
    def __init__(self, model_path, num_classes=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        hidden_size = self.encoder.config.hidden_size
        self.token_attention_layer = nn.Linear(hidden_size, 1)  # Token-level attention
        self.classifier = nn.Linear(hidden_size*2, num_classes)
    
    def token_attention_pool(self, last_hidden_state, attention_mask):
        # last_hidden_state: (B, T, H)

        # 各tokenにスコアを付与
        scores = self.token_attention_layer(last_hidden_state).squeeze(-1)  # (B, T)

        # paddingを除外
        scores = scores.masked_fill(attention_mask == 0, -1e9)

        # softmaxで重要度に変換
        token_attention = torch.softmax(scores, dim=1)  # (B, T)

        # 重み付き平均
        pooled = torch.sum(
            last_hidden_state * token_attention.unsqueeze(-1),
            dim=1
        )  # (B, H)

        return pooled, token_attention
    
    # Encode (Ex. DeBERTa)
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    # Forward pass
    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        feat_a = self.encode(input_ids_a, attention_mask_a) # (B, T, H)
        feat_a, _ = self.token_attention_pool(feat_a.last_hidden_state, attention_mask_a) # (B, H)

        feat_b = self.encode(input_ids_b, attention_mask_b) # (B, T, H)
        feat_b, _ = self.token_attention_pool(feat_b.last_hidden_state, attention_mask_b) # (B, H)

        feats = torch.cat([feat_a, feat_b], dim=1) # (B, H*2)
  
        logits = self.classifier(feats)

        return logits.float()