from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleTfidfReranker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))

    def rerank(self, query: str, docs: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        if not docs:
            return docs
        texts = [d.get('text', '') for d in docs]
        X = self.vectorizer.fit_transform(texts + [query])
        D = X[:-1]
        q = X[-1]
        sims = cosine_similarity(D, q)
        scores = sims.ravel()
        order = np.argsort(scores)[::-1]
        reranked: List[Dict] = []
        for idx in order:
            item = dict(docs[int(idx)])
            item['score'] = float(scores[int(idx)])
            reranked.append(item)
        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked


