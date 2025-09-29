import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List


class HFTextEmbedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state  # [B, T, H]
            attention_mask = inputs['attention_mask']    # [B, T]
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            masked = token_embeddings * mask
            sum_embeddings = masked.sum(dim=1)
            lengths = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
            embeddings = sum_embeddings / lengths
            emb_np = embeddings.cpu().numpy().astype('float32')
            if normalize:
                norms = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-12
                emb_np = emb_np / norms
            all_embeddings.append(emb_np)
        return np.vstack(all_embeddings) if all_embeddings else np.zeros((0, self.model.config.hidden_size), dtype='float32')


