import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def embed_text_column(text_series: pd.Series) -> np.ndarray:
    """
    Converts a pandas Series of text into BERT CLS embeddings.
    """
    embeddings = []
    for text in text_series:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)

        token_embeddings = outputs.last_hidden_state.squeeze(0)  # shape: [seq_len, 768]
        sentence_embedding = token_embeddings.mean(dim=0).numpy()  # shape: [768]
        embeddings.append(sentence_embedding)
    return np.vstack(embeddings)

def build_features(X: pd.DataFrame) -> np.ndarray:
    """
    Combines structured and embedded features.
    """
    structured = X.drop(columns=["narrative"], errors="ignore").to_numpy()
    if "narrative" in X.columns:
        embedded = embed_text_column(X["narrative"])
        return np.hstack([structured, embedded])
    return structured
