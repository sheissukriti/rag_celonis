import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
import yaml
from rag.retrieve import get_retriever
from rag.chain import build_prompt, ContextDoc
import httpx


@st.cache_resource
def load_pipeline():
    with open('configs/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    retriever_kind = cfg['app'].get('retriever', 'bm25')
    qa_dump_path = (
        cfg['app']['qa_texts_path'] if retriever_kind in ('bm25', 'faiss_tfidf') else cfg['app']['index_path'].replace('.index', '.qa.jsonl')
    )
    ids_path_opt = cfg['app']['index_path'].replace('.index', '.ids') if retriever_kind in ('dense', 'faiss_tfidf') else None
    emb_path_opt = cfg['app']['index_path'].replace('.index', '.emb.npy') if retriever_kind == 'dense' else None
    retriever = get_retriever(
        kind=retriever_kind,
        index_path=cfg['app']['index_path'],
        embedding_model=cfg['app']['embedding_model'],
        qa_dump_path=qa_dump_path,
        embeddings_path=emb_path_opt,
        ids_path=ids_path_opt,
    )
    return cfg, retriever


def main():
    st.set_page_config(page_title='Customer Support RAG Assistant', page_icon='🤖', layout='centered')
    st.title('Customer Support RAG Assistant')
    st.caption('Ask a question. Top contexts and answer will be shown with citations.')

    cfg, retriever = load_pipeline()
    top_k = st.sidebar.number_input('Retrieve top_k', min_value=1, max_value=50, value=cfg['app']['top_k'])
    top_k_context = st.sidebar.number_input('Context count', min_value=1, max_value=10, value=cfg['app']['top_k_context'])

    query = st.text_input('Your question')
    if st.button('Get answer') and query.strip():
        with st.spinner('Retrieving...'):
            retrieved = retriever.search(query, top_k=int(top_k))
            contexts = [ContextDoc(**r) for r in retrieved[: int(top_k_context)]]
            prompt = build_prompt(query, contexts)
        # Try Ollama; fallback placeholder
        try:
            with httpx.Client(timeout=30) as client:
                r = client.post('http://127.0.0.1:11434/api/generate', json={
                    'model': cfg['app'].get('model', 'mistral'),
                    'prompt': prompt,
                    'stream': False,
                    'options': {'num_predict': 256}
                })
                r.raise_for_status()
                data = r.json()
                answer = data.get('response', '') or '(empty response)'
        except Exception:
            answer = '(LLM placeholder) Ollama not reachable. Start ollama and pull a model (e.g., mistral).'

        st.subheader('Answer')
        st.write(answer)
        st.subheader('Citations')
        for i, c in enumerate(contexts, 1):
            with st.expander(f'Doc {i}  •  score={c.score:.4f}'):
                st.write(c.text)


if __name__ == '__main__':
    main()


