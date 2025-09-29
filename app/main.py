
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import yaml
from rag.retrieve import get_retriever
from rag.chain import build_prompt, ContextDoc
import httpx
from rag.rerank import SimpleTfidfReranker

# NOTE: Replace this LLM stub with your preferred model (e.g., ollama, OpenAI, vLLM)
class OllamaLLM:
    def __init__(self, model: str = 'mistral'):
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            with httpx.Client(timeout=30) as client:
                r = client.post('http://127.0.0.1:11434/api/generate', json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {'num_predict': max_tokens}
                })
                r.raise_for_status()
                data = r.json()
                return data.get('response', '') or '(empty response)'
        except Exception:
            return '(LLM placeholder) Ollama not reachable. Please start ollama and pull a model (e.g., mistral).' 

app = FastAPI()

with open('configs/config.yaml', 'r') as f:
    CFG = yaml.safe_load(f)

retriever_kind = CFG['app'].get('retriever', 'bm25')
qa_dump_path = (
    CFG['app']['qa_texts_path'] if retriever_kind in ('bm25', 'faiss_tfidf') else CFG['app']['index_path'].replace('.index', '.qa.jsonl')
)
ids_path_opt = CFG['app']['index_path'].replace('.index', '.ids') if retriever_kind in ('dense', 'faiss_tfidf') else None
emb_path_opt = CFG['app']['index_path'].replace('.index', '.emb.npy') if retriever_kind == 'dense' else None
RETRIEVER = get_retriever(
    kind=retriever_kind,
    index_path=CFG['app']['index_path'],
    embedding_model=CFG['app']['embedding_model'],
    qa_dump_path=qa_dump_path,
    embeddings_path=emb_path_opt,
    ids_path=ids_path_opt,
)
RERANKER = SimpleTfidfReranker() if CFG['app'].get('use_reranker', False) else None
LLM = OllamaLLM(CFG['app'].get('model', 'mistral'))

class GenerateRequest(BaseModel):
    query: str
    top_k: int | None = None
    top_k_context: int | None = None
    max_tokens: int | None = 256

class Citation(BaseModel):
    id: int
    score: float
    text: str

class GenerateResponse(BaseModel):
    answer: str
    citations: List[Citation]

@app.post('/generate_response', response_model=GenerateResponse)
async def generate_response(req: GenerateRequest):
    top_k = req.top_k or CFG['app']['top_k']
    top_k_context = req.top_k_context or CFG['app']['top_k_context']

    retrieved = RETRIEVER.search(req.query, top_k=top_k)
    if RERANKER is not None:
        retrieved = RERANKER.rerank(req.query, retrieved, top_k=top_k)
    contexts = [ContextDoc(**r) for r in retrieved[:top_k_context]]
    prompt = build_prompt(req.query, contexts)

    answer = LLM.generate(prompt, max_tokens=req.max_tokens or 256)
    return GenerateResponse(
        answer=answer,
        citations=[Citation(id=c.id, score=c.score, text=c.text) for c in contexts],
    )
