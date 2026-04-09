from .model import PairwiseDebertaClassifier, DebertaClassifier_TokenAttentionPooling

MODEL_DICT = {
    "Baseline": PairwiseDebertaClassifier,
    "TokenAttention": DebertaClassifier_TokenAttentionPooling
}