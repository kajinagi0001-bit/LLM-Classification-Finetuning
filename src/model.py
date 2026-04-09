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
        #print(f"feats: {feats[0]}")
        logits = self.classifier(feats)
        #print(f"logits: {logits[0]}")
        return logits.float()