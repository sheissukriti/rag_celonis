
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import yaml
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from rag.retrieve import get_retriever
from rag.chain import build_prompt, ContextDoc
import httpx
from rag.rerank import SimpleTfidfReranker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

# Optional evaluation import (fallback if dependencies missing)
try:
    from rag.evaluation import RAGEvaluator, create_test_queries
    EVALUATION_AVAILABLE = True
    logger.info("Evaluation module loaded successfully")
except ImportError as e:
    logger.warning(f"Evaluation module not available: {e}")
    EVALUATION_AVAILABLE = False
    
    # Fallback classes
    class RAGEvaluator:
        def __init__(self, *args, **kwargs):
            pass
        def evaluate_batch(self, data):
            return {"error": "Evaluation not available - missing dependencies"}
    
    def create_test_queries():
        return [
            {"query": "I need help with my order"},
            {"query": "How do I reset my password?"},
            {"query": "My product arrived damaged"},
            {"query": "Can I return this item?"},
            {"query": "How long does shipping take?"}
        ]

class OllamaLLM:
    """LLM integration with Ollama. Supports fallback to OpenAI-compatible APIs."""
    
    def __init__(self, model: str = 'mistral', base_url: str = 'http://127.0.0.1:11434'):
        self.model = model
        self.base_url = base_url
        logger.info(f"Initialized OllamaLLM with model: {model}")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate response using Ollama API with proper error handling."""
        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(f'{self.base_url}/api/generate', json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_predict': max_tokens,
                        'temperature': temperature
                    }
                })
                response.raise_for_status()
                data = response.json()
                
                generated_text = data.get('response', '').strip()
                if not generated_text:
                    logger.warning("Empty response from LLM")
                    return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
                logger.info(f"Successfully generated response of length: {len(generated_text)}")
                return generated_text
                
        except httpx.TimeoutException:
            error_msg = "Request timed out. The LLM service may be overloaded."
            logger.error(f"Timeout error: {error_msg}")
            return f"Error: {error_msg}"
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(f"HTTP error: {error_msg}")
            return f"Error: LLM service returned an error. Please try again."
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"LLM generation error: {error_msg}")
            return "Error: LLM service is not available. Please ensure Ollama is running and the model is installed."

app = FastAPI(
    title="Customer Support RAG Assistant",
    description="A Retrieval-Augmented Generation assistant for customer support queries",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration with error handling
try:
    with open('configs/config.yaml', 'r') as f:
        CFG = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
except FileNotFoundError:
    logger.error("Configuration file not found. Please ensure configs/config.yaml exists.")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error parsing configuration file: {e}")
    raise

# Initialize components with error handling
try:
    retriever_kind = CFG['app'].get('retriever', 'bm25')
    qa_dump_path = (
        CFG['app']['qa_texts_path'] if retriever_kind in ('bm25', 'faiss_tfidf') 
        else CFG['app']['index_path'].replace('.index', '.qa.jsonl')
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
    logger.info(f"Initialized {retriever_kind} retriever successfully")
    
    RERANKER = SimpleTfidfReranker() if CFG['app'].get('use_reranker', False) else None
    if RERANKER:
        logger.info("Initialized TF-IDF reranker")
    
    LLM = OllamaLLM(CFG['app'].get('model', 'mistral'))
    
    # Initialize evaluator (with fallback)
    if EVALUATION_AVAILABLE:
        EVALUATOR = RAGEvaluator(CFG['app']['embedding_model'])
    else:
        EVALUATOR = RAGEvaluator()  # Fallback evaluator
    
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise

def log_response(query: str, answer: str, citations: List[dict], response_time: float):
    """Log responses to file for analysis and debugging."""
    if not CFG['app'].get('log_responses', False):
        return
        
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer': answer,
        'citations': citations,
        'response_time_seconds': response_time,
        'retriever_type': retriever_kind,
        'model': CFG['app'].get('model', 'mistral')
    }
    
    log_file = CFG['app'].get('log_file', 'logs/responses.jsonl')
    Path(log_file).parent.mkdir(exist_ok=True)
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"Failed to log response: {e}")

class GenerateRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    top_k_context: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class Citation(BaseModel):
    id: int
    score: float
    text: str

class GenerateResponse(BaseModel):
    answer: str
    citations: List[Citation]
    response_time_seconds: float
    retriever_type: str
    query_processed: str

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Customer Support RAG Assistant API",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check including component status."""
    try:
        # Test retriever
        test_results = RETRIEVER.search("test query", top_k=1)
        retriever_status = "healthy" if test_results is not None else "error"
    except Exception:
        retriever_status = "error"
    
    return {
        "status": "healthy",
        "components": {
            "retriever": retriever_status,
            "reranker": "enabled" if RERANKER else "disabled",
            "llm": "configured"
        },
        "config": {
            "retriever_type": retriever_kind,
            "model": CFG['app'].get('model', 'mistral')
        }
    }

@app.post('/generate_response', response_model=GenerateResponse)
async def generate_response(req: GenerateRequest):
    """Generate a response to a customer support query using RAG."""
    start_time = datetime.now()
    
    try:
        # Validate input
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        query = req.query.strip()
        top_k = req.top_k or CFG['app']['top_k']
        top_k_context = req.top_k_context or CFG['app']['top_k_context']
        max_tokens = req.max_tokens or CFG['app']['max_tokens']
        temperature = req.temperature or CFG['app']['temperature']
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Retrieve relevant documents
        retrieved = RETRIEVER.search(query, top_k=top_k)
        if not retrieved:
            logger.warning("No documents retrieved for query")
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Apply reranking if enabled
        if RERANKER is not None:
            retrieved = RERANKER.rerank(query, retrieved, top_k=top_k)
            logger.info(f"Reranked {len(retrieved)} documents")
        
        # Prepare context and generate response
        contexts = [ContextDoc(**r) for r in retrieved[:top_k_context]]
        prompt = build_prompt(query, contexts)
        
        answer = LLM.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = GenerateResponse(
            answer=answer,
            citations=[Citation(id=c.id, score=c.score, text=c.text) for c in contexts],
            response_time_seconds=response_time,
            retriever_type=retriever_kind,
            query_processed=query
        )
        
        # Log response
        log_response(
            query=query,
            answer=answer,
            citations=[{"id": c.id, "score": c.score, "text": c.text[:200]} for c in contexts],
            response_time=response_time
        )
        
        logger.info(f"Generated response in {response_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/evaluate")
async def evaluate_system():
    """Evaluate the RAG system using predefined test queries."""
    if not EVALUATION_AVAILABLE:
        return {
            "status": "error",
            "message": "Evaluation system not available - missing dependencies (likely _lzma module)",
            "suggestion": "Install xz-utils and rebuild Python, or use Docker deployment"
        }
    
    try:
        test_queries = create_test_queries()
        evaluation_data = []
        
        logger.info(f"Starting evaluation with {len(test_queries)} test queries")
        
        for test_query in test_queries:
            start_time = datetime.now()
            
            # Generate response using the same logic as generate_response
            query = test_query["query"]
            retrieved = RETRIEVER.search(query, top_k=CFG['app']['top_k'])
            
            if RERANKER is not None:
                retrieved = RERANKER.rerank(query, retrieved, top_k=CFG['app']['top_k'])
            
            contexts = [ContextDoc(**r) for r in retrieved[:CFG['app']['top_k_context']]]
            prompt = build_prompt(query, contexts)
            answer = LLM.generate(prompt, max_tokens=CFG['app']['max_tokens'])
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            evaluation_data.append({
                'query': query,
                'answer': answer,
                'citations': [{'id': c.id, 'score': c.score, 'text': c.text} for c in contexts],
                'response_time': response_time
            })
        
        # Run evaluation
        aggregate_metrics = EVALUATOR.evaluate_batch(evaluation_data)
        
        logger.info("Evaluation completed successfully")
        return {
            "status": "completed",
            "metrics": aggregate_metrics,
            "test_queries_count": len(test_queries),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail="Evaluation failed")

@app.get("/test-queries")
async def get_test_queries():
    """Get the list of test queries used for evaluation."""
    return {
        "test_queries": create_test_queries(),
        "count": len(create_test_queries())
    }
