
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from .embedder import HFTextEmbedder
import numpy as np
from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(self, index_path: str, embedding_model: str, qa_dump_path: str):
        self.model = HFTextEmbedder(embedding_model)
        self.index = faiss.read_index(index_path)
        self.ids = [int(x) for x in Path(index_path).with_suffix('.ids').read_text().splitlines()]
        self.qa_map: Dict[int, str] = {}
        with open(qa_dump_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.qa_map[int(obj['id'])] = obj['text']

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        q = self.model.encode([query], normalize=True)
        q = np.asarray(q, dtype='float32')
        scores, idxs = self.index.search(q, top_k)
        results: List[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            doc_id = self.ids[idx]
            results.append({
                'id': int(doc_id),
                'score': float(score),
                'text': self.qa_map.get(doc_id, '')
            })
        return results


class BM25Retriever:
    def __init__(self, qa_dump_path: str):
        self.docs: List[str] = []
        self.ids: List[int] = []
        with open(qa_dump_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.ids.append(int(obj['id']))
                self.docs.append(obj['text'])
        tokenized = [self._tokenize(d) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        scores = self.bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        results: List[dict] = []
        for idx in top_idx:
            results.append({
                'id': int(self.ids[idx]),
                'score': float(scores[idx]),
                'text': self.docs[idx]
            })
        return results


class DenseRetriever:
    def __init__(self, embedding_model: str, embeddings_path: str, ids_path: str, qa_dump_path: str):
        self.model = HFTextEmbedder(embedding_model)
        self.embeddings = np.load(embeddings_path).astype('float32')
        self.ids = [int(x) for x in Path(ids_path).read_text().splitlines()]
        self.qa_map: Dict[int, str] = {}
        with open(qa_dump_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.qa_map[int(obj['id'])] = obj['text']

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        q = self.model.encode([query], normalize=True).astype('float32')
        # cosine since normalized => dot product
        sims = (self.embeddings @ q.T).ravel()
        top_idx = np.argsort(sims)[::-1][:top_k]
        results: List[dict] = []
        for idx in top_idx:
            doc_id = self.ids[int(idx)]
            results.append({'id': int(doc_id), 'score': float(sims[int(idx)]), 'text': self.qa_map.get(doc_id, '')})
        return results


class FaissTfidfRetriever:
    def __init__(self, index_path: str, ids_path: str, qa_dump_path: str):
        self.index = faiss.read_index(index_path)
        self.ids = [int(x) for x in Path(ids_path).read_text().splitlines()]
        self.qa_map: Dict[int, str] = {}
        with open(qa_dump_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.qa_map[int(obj['id'])] = obj['text']
        from joblib import load
        self.vec = load(Path(index_path).with_suffix('.tfidf.pkl'))
        self.svd = load(Path(index_path).with_suffix('.svd.pkl'))

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        Xq = self.vec.transform([query])
        z = self.svd.transform(Xq)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        z = np.asarray(z, dtype='float32')
        scores, idxs = self.index.search(z, top_k)
        results: List[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            doc_id = self.ids[idx]
            results.append({'id': int(doc_id), 'score': float(score), 'text': self.qa_map.get(doc_id, '')})
        return results


def get_retriever(kind: str, index_path: Optional[str], embedding_model: str, qa_dump_path: str, embeddings_path: Optional[str] = None, ids_path: Optional[str] = None):
    if kind == 'bm25':
        return BM25Retriever(qa_dump_path)
    if kind == 'dense':
        assert embeddings_path and ids_path, 'embeddings_path and ids_path required for dense retriever'
        return DenseRetriever(embedding_model, embeddings_path, ids_path, qa_dump_path)
    if kind == 'faiss_tfidf':
        assert index_path and ids_path, 'index_path and ids_path required for faiss_tfidf retriever'
        return FaissTfidfRetriever(index_path, ids_path, qa_dump_path)
    return Retriever(index_path, embedding_model, qa_dump_path)
