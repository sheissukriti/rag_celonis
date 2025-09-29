
import json
import faiss
import numpy as np
from pathlib import Path
from .embedder import HFTextEmbedder


def load_texts(path: str) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    texts: list[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            ids.append(int(obj['id']))
            texts.append(obj['text'])
    return ids, texts


def build_index(texts: list[str], model_name: str) -> tuple[faiss.Index, np.ndarray]:
    embedder = HFTextEmbedder(model_name)
    emb = embedder.encode(texts, batch_size=64, normalize=True)
    emb = np.asarray(emb, dtype='float32')
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, emb


def save_index(index: faiss.Index, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, path)


def main(input_path: str, qa_path: str, index_path: str, embedding_model: str, save_embeddings: bool = True) -> None:
    ids_in, texts_in = load_texts(input_path)
    ids_qa, texts_qa = load_texts(qa_path)
    assert ids_in == ids_qa
    index, emb = build_index(texts_in, embedding_model)
    save_index(index, index_path)
    (Path(index_path).with_suffix('.ids')).write_text('\n'.join(map(str, ids_in)), encoding='utf-8')
    (Path(index_path).with_suffix('.qa.jsonl')).write_text(open(qa_path, 'r', encoding='utf-8').read(), encoding='utf-8')
    if save_embeddings:
        np.save(Path(index_path).with_suffix('.emb.npy'), emb)


if __name__ == '__main__':
    import argparse, yaml
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='store/input_texts.jsonl')
    p.add_argument('--qa', default='store/qa_texts.jsonl')
    p.add_argument('--index', default='store/faiss.index')
    p.add_argument('--embedding_model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--no_save_embeddings', action='store_true')
    args = p.parse_args()
    main(args.input, args.qa, args.index, args.embedding_model, save_embeddings=not args.no_save_embeddings)
