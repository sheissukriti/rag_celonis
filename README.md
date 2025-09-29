# Customer Support RAG Assistant

## Overview
A Retrieval-Augmented Generation (RAG) assistant for customer support, built over the Twitter Customer Support dataset. It provides:
- Retrieval using FAISS + TF‑IDF‑SVD (default) or BM25, with optional TF‑IDF reranking
- FastAPI API endpoint for programmatic access
- Streamlit UI for interactive querying
- Optional Ollama integration as the LLM backend

## Why these choices
- Retrieval backends:
  - FAISS + TF‑IDF‑SVD: dense‑ish vectors without GPU/heavy model memory; fast and lightweight.
  - BM25 (+ TF‑IDF rerank): robust classic baseline; no model weights required.
- Ollama: easy local LLM integration; replaceable with any API (OpenAI, vLLM) later.
- Subsetting (10k) indices for fast local iteration; can scale to more documents with the same scripts.
- Explainability: citations (id, score, text) are returned alongside answers.

## Architecture
```mermaid
flowchart TD
    A[User Query] -->|POST /generate_response| B[FastAPI]
    A -->|Streamlit UI| S[Streamlit App]
    S -->|uses| B
    B --> C[Retriever]
    C -->|Top-K docs| D[Prompt Builder]
    D --> E[LLM (Ollama)]
    E --> F[Answer]
    C --> G[Citations]
    F --> H[Response JSON]
    G --> H
    H -->|display| S

    subgraph Retrieval Options
      C1[FAISS + TF-IDF-SVD]
      C2[BM25]
      C3[TF-IDF Reranker]
    end

    C <--> C1
    C <--> C2
    C --> C3
```

## Repo layout
- `app/main.py`: FastAPI server and Ollama integration
- `app/streamlit_app.py`: Streamlit UI
- `rag/ingest.py`: dataset ingestion to JSONL (uses local copy or datasets-server)
- `rag/index.py`: dense index builder (with HuggingFace embeddings; optional)
- `rag/index_tfidf.py`: TF‑IDF‑SVD embeddings + FAISS index builder
- `rag/retrieve.py`: retrievers (BM25, dense, FAISS‑TFIDF, FAISS)
- `rag/rerank.py`: TF‑IDF reranker
- `rag/chain.py`: prompt assembly with citations
- `configs/config.yaml`: runtime configuration
- `store/`: generated artifacts (jsonl, faiss, ids, etc.)

## Prerequisites
- macOS with Homebrew (for xz) and pyenv recommended
- Python 3.12 (built with lzma); this project uses a local `.venv`
- Optional: Ollama running locally for LLM responses

## Setup
1) Clone and enter the project directory, then create venv
```bash
cd "<path>/rag_c"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) (If lzma missing) Install xz and rebuild Python via pyenv
```bash
brew install xz
# Ensure pyenv up to date, then install
pyenv install 3.12.7
pyenv local 3.12.7
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Download dataset snapshot (done once)
```bash
python - << 'PY'
import requests
from pathlib import Path
rev = 'edd299bba12cb076030822ea97b4329d6041f218'
url = f"https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k/resolve/{rev}/preprocessed_data.json?download=true"
Path('data').mkdir(exist_ok=True)
r = requests.get(url, stream=True, timeout=120)
r.raise_for_status()
with open('data/preprocessed_data.json', 'wb') as f:
    for chunk in r.iter_content(1024*1024):
        if chunk:
            f.write(chunk)
print('Downloaded data/preprocessed_data.json')
PY
```

4) Ingest a 50k subset to JSONL (adjust `limit` as desired)
```bash
python - << 'PY'
from rag.ingest import main
main('store/input_texts.jsonl','store/qa_texts.jsonl', limit=50000)
print('Ingestion complete')
PY
```

5) Build TF‑IDF‑SVD FAISS index over a 10k subset (fast dev loop)
```bash
head -n 10000 store/input_texts.jsonl > store/input_texts_10k.jsonl
head -n 10000 store/qa_texts.jsonl > store/qa_texts_10k.jsonl
python - << 'PY'
from rag.index_tfidf import main
main('store/input_texts_10k.jsonl','store/qa_texts_10k.jsonl','store/faiss.index', n_components=256)
print('Built TFIDF+SVD FAISS (10k)')
PY
```

6) Configure runtime in `configs/config.yaml`
```yaml
app:
  model: mistral
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  top_k: 10
  top_k_context: 5
  use_reranker: true
  index_path: store/faiss.index
  input_texts_path: store/input_texts.jsonl
  qa_texts_path: store/qa_texts_10k.jsonl
  retriever: faiss_tfidf   # options: bm25 | faiss_tfidf | dense | faiss
```

7) (Optional) Start Ollama and pull a model
```bash
ollama serve
ollama pull mistral
```

8) Run API
```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Test:
```bash
curl -s -X POST http://127.0.0.1:8000/generate_response \
  -H 'Content-Type: application/json' \
  -d '{"query":"I need help resetting my password. I didn’t receive the reset link.", "top_k": 10, "top_k_context": 3 }'
```

9) Run Streamlit UI
```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```
Open http://127.0.0.1:8501

## Notes on decisions
- lzma error workaround: installed xz and rebuilt Python via pyenv to ensure `datasets` works.
- Memory constraints: provided non‑neural TF‑IDF‑SVD as a dense retrieval path and BM25 as a no‑model fallback.
- Reranking: TF‑IDF reranker improves top‑k ordering without heavy model costs.
- Explainability: citations with scores and text are returned for transparency.

## Next steps
- Swap placeholder generation with a hosted LLM for higher quality
- Add evaluation harness (Recall@K, ROUGE‑L, LLM‑as‑judge) scripts
- Scale to larger subsets or full dataset with IVF/HNSW indices
