import json
from pathlib import Path
from typing import Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import faiss


def load_texts(path: str) -> Tuple[list[int], list[str]]:
    ids, texts = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            o = json.loads(line)
            ids.append(int(o['id']))
            texts.append(o['text'])
    return ids, texts


def build_tfidf_svd(texts: list[str], n_components: int = 256):
    vec = TfidfVectorizer(max_features=200000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z = svd.fit_transform(X)
    # L2 normalize
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z.astype('float32'), vec, svd


def main(input_path: str, qa_path: str, index_path: str, n_components: int = 256) -> None:
    ids_in, texts_in = load_texts(input_path)
    ids_qa, _ = load_texts(qa_path)
    assert ids_in == ids_qa
    emb, vec, svd = build_tfidf_svd(texts_in, n_components=n_components)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    (Path(index_path).with_suffix('.ids')).write_text('\n'.join(map(str, ids_in)), encoding='utf-8')
    (Path(index_path).with_suffix('.emb.npy')).write_bytes(emb.tobytes())
    joblib.dump(vec, Path(index_path).with_suffix('.tfidf.pkl'))
    joblib.dump(svd, Path(index_path).with_suffix('.svd.pkl'))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='store/input_texts.jsonl')
    p.add_argument('--qa', default='store/qa_texts.jsonl')
    p.add_argument('--index', default='store/faiss.index')
    p.add_argument('--n_components', type=int, default=256)
    args = p.parse_args()
    main(args.input, args.qa, args.index, args.n_components)


