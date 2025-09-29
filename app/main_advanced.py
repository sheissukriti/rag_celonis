"""
Enhanced FastAPI main application integrating all RAG system improvements.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yaml
import logging
import json
import os
from datetime import datetime
from pathlib import Path

# Core RAG imports
from rag.retrieve import get_retriever
from rag.chain import build_prompt, ContextDoc
from rag.rerank import SimpleTfidfReranker
import httpx

# New feature imports
from rag.conversation import ConversationManager, ContextAwarePromptBuilder
from rag.advanced_rerank import create_advanced_reranker, RERANKER_CONFIGS
from rag.cache import CacheManager, ResponseCache, DEFAULT_CACHE_CONFIGS
from rag.ab_testing import ExperimentManager, ABTestingService, EXAMPLE_EXPERIMENTS
from rag.feedback_learning import RealTimeLearningSystem, FeedbackType
from rag.multilingual import MultilingualRAGSystem, create_multilingual_system

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

# Ensure directories exist
for directory in ['logs', 'store', 'store/conversations', 'store/experiments', 'store/feedback', 'locales']:
    Path(directory).mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(
    title="Advanced RAG Customer Support Assistant",
    description="Enhanced RAG system with multi-turn conversations, advanced reranking, caching, A/B testing, real-time learning, and multi-language support",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
try:
    with open('configs/config.yaml', 'r') as f:
        CFG = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise

# Enhanced LLM class
class OllamaLLM:
    """Enhanced LLM with better error handling and features."""
    
    def __init__(self, model: str = 'mistral', base_url: str = 'http://127.0.0.1:11434'):
        self.model = model
        self.base_url = base_url
        logger.info(f"Initialized OllamaLLM with model: {model}")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate response using Ollama API."""
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
                
                return generated_text
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "Error: LLM service is not available. Please ensure Ollama is running and the model is installed."

# Initialize components
try:
    # Basic components
    retriever_kind = CFG['app'].get('retriever', 'bm25')
    qa_dump_path = (
        CFG['app']['qa_texts_path'] if retriever_kind in ('bm25', 'faiss_tfidf') 
        else CFG['app']['index_path'].replace('.index', '.qa.jsonl')
    )
    ids_path_opt = CFG['app']['index_path'].replace('.index', '.ids') if retriever_kind in ('dense', 'faiss_tfidf') else None
    emb_path_opt = CFG['app']['index_path'].replace('.index', '.emb.npy') if retriever_kind == 'dense' else None
    
    BASE_RETRIEVER = get_retriever(
        kind=retriever_kind,
        index_path=CFG['app']['index_path'],
        embedding_model=CFG['app']['embedding_model'],
        qa_dump_path=qa_dump_path,
        embeddings_path=emb_path_opt,
        ids_path=ids_path_opt,
    )
    
    BASE_LLM = OllamaLLM(CFG['app'].get('model', 'mistral'))
    
    # Enhanced components
    CONVERSATION_MANAGER = ConversationManager()
    CONTEXT_AWARE_PROMPT_BUILDER = ContextAwarePromptBuilder(CONVERSATION_MANAGER)
    
    # Advanced reranker
    ADVANCED_RERANKER = create_advanced_reranker(RERANKER_CONFIGS['cross_encoder'])
    
    # Caching system
    cache_config = DEFAULT_CACHE_CONFIGS.get('redis', DEFAULT_CACHE_CONFIGS['memory'])
    CACHE_MANAGER = CacheManager(cache_config)
    RESPONSE_CACHE = ResponseCache(CACHE_MANAGER)
    
    # A/B testing
    EXPERIMENT_MANAGER = ExperimentManager()
    AB_TESTING_SERVICE = ABTestingService(EXPERIMENT_MANAGER)
    
    # Real-time learning
    LEARNING_SYSTEM = RealTimeLearningSystem(BASE_RETRIEVER)
    
    # Multilingual system
    MULTILINGUAL_SYSTEM = create_multilingual_system(BASE_RETRIEVER, BASE_LLM)
    
    logger.info("All enhanced components initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize enhanced components: {e}")
    raise

# Pydantic models
class EnhancedGenerateRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    language: Optional[str] = 'en'
    auto_translate: Optional[bool] = True
    top_k: Optional[int] = None
    top_k_context: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    use_caching: Optional[bool] = True
    use_advanced_reranking: Optional[bool] = True
    enable_learning: Optional[bool] = True
    query_intent: Optional[str] = None
    context_boost: Optional[float] = 1.0

class FeedbackRequest(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query: str
    response: str
    citations: List[Dict[str, Any]]
    feedback_type: str
    feedback_value: Any
    metadata: Optional[Dict[str, Any]] = None

class ExperimentCreateRequest(BaseModel):
    name: str
    description: str
    variants: List[Dict[str, Any]]
    metrics: List[Dict[str, Any]]
    created_by: Optional[str] = None

# API Endpoints
@app.get("/")
async def root():
    """Enhanced root endpoint."""
    return {
        "message": "Advanced RAG Customer Support Assistant API",
        "status": "healthy",
        "version": "2.0.0",
        "features": [
            "multi_turn_conversations",
            "advanced_reranking", 
            "redis_caching",
            "ab_testing",
            "real_time_learning",
            "multi_language_support",
            "analytics_dashboard"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with all components."""
    try:
        # Test base retriever
        test_results = BASE_RETRIEVER.search("test query", top_k=1)
        retriever_status = "healthy" if test_results is not None else "error"
    except Exception:
        retriever_status = "error"
    
    # Check cache
    cache_stats = CACHE_MANAGER.get_stats()
    cache_status = "healthy" if cache_stats.get('connected', True) else "error"
    
    # Check learning system
    learning_stats = LEARNING_SYSTEM.get_system_stats()
    learning_status = "healthy"
    
    # Check multilingual
    ml_stats = MULTILINGUAL_SYSTEM.get_language_stats()
    ml_status = "healthy" if ml_stats.get('language_detector_available', False) else "limited"
    
    return {
        "status": "healthy",
        "components": {
            "base_retriever": retriever_status,
            "advanced_reranker": "enabled",
            "cache": cache_status,
            "conversations": "enabled",
            "ab_testing": "enabled", 
            "learning_system": learning_status,
            "multilingual": ml_status,
            "llm": "configured"
        },
        "config": {
            "retriever_type": retriever_kind,
            "model": CFG['app'].get('model', 'mistral'),
            "cache_type": cache_stats.get('type', 'unknown'),
            "languages_supported": len(MULTILINGUAL_SYSTEM.translation_service.get_supported_languages())
        },
        "stats": {
            "cache": cache_stats,
            "learning": learning_stats,
            "multilingual": ml_stats
        }
    }

@app.post('/generate_response_advanced')
async def generate_response_advanced(req: EnhancedGenerateRequest, background_tasks: BackgroundTasks):
    """Enhanced response generation with all features."""
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
        
        logger.info(f"Processing enhanced query: {query[:100]}...")
        
        # A/B testing - get configuration
        ab_config = AB_TESTING_SERVICE.get_retrieval_config(req.user_id, req.conversation_id)
        experiment_variant = ab_config.get('_variant_id')
        
        # Check cache first
        cache_key_params = {
            'top_k': top_k,
            'top_k_context': top_k_context,
            'language': req.language,
            'variant': experiment_variant
        }
        
        cached_response = None
        if req.use_caching:
            cached_response = RESPONSE_CACHE.get_response(query, cache_key_params)
        
        if cached_response:
            logger.info("Cache hit - returning cached response")
            cached_response['cache_hit'] = True
            return cached_response
        
        # Multilingual processing
        if req.language and req.language != 'en':
            ml_response = MULTILINGUAL_SYSTEM.process_query(
                query, req.language, top_k
            )
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Cache the response
            if req.use_caching:
                background_tasks.add_task(
                    RESPONSE_CACHE.cache_response,
                    query, cache_key_params, ml_response
                )
            
            # Record A/B testing metrics
            if experiment_variant:
                background_tasks.add_task(
                    AB_TESTING_SERVICE.record_interaction,
                    query, response_time, 0.8, None,
                    req.user_id, req.conversation_id, ab_config
                )
            
            ml_response['response_time_seconds'] = response_time
            ml_response['cache_hit'] = False
            ml_response['experiment_variant'] = experiment_variant
            
            return ml_response
        
        # Use learning system's adaptive retriever
        if req.enable_learning:
            retrieved = LEARNING_SYSTEM.search(query, top_k)
        else:
            retrieved = BASE_RETRIEVER.search(query, top_k)
        
        if not retrieved:
            logger.warning("No documents retrieved for query")
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Advanced reranking
        if req.use_advanced_reranking:
            retrieved = ADVANCED_RERANKER.rerank(query, retrieved, top_k)
            logger.info(f"Advanced reranked {len(retrieved)} documents")
        
        # Context-aware prompt building
        contexts = [ContextDoc(**r) for r in retrieved[:top_k_context]]
        
        if req.conversation_id:
            prompt = CONTEXT_AWARE_PROMPT_BUILDER.build_context_aware_prompt(
                query, [r.__dict__ for r in contexts], req.conversation_id
            )
        else:
            prompt = build_prompt(query, contexts)
        
        # Generate response
        answer = BASE_LLM.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Add to conversation if conversation_id provided
        if req.conversation_id:
            CONVERSATION_MANAGER.add_turn(
                req.conversation_id, query, answer,
                [{"id": c.id, "score": c.score, "text": c.text} for c in contexts]
            )
        
        # Prepare response
        response = {
            "answer": answer,
            "citations": [{"id": c.id, "score": c.score, "text": c.text} for c in contexts],
            "response_time_seconds": response_time,
            "retriever_type": retriever_kind,
            "query_processed": query,
            "cache_hit": False,
            "experiment_variant": experiment_variant,
            "advanced_reranking_used": req.use_advanced_reranking,
            "learning_enabled": req.enable_learning,
            "conversation_id": req.conversation_id
        }
        
        # Cache the response
        if req.use_caching:
            background_tasks.add_task(
                RESPONSE_CACHE.cache_response,
                query, cache_key_params, response
            )
        
        # Record A/B testing metrics
        if experiment_variant:
            background_tasks.add_task(
                AB_TESTING_SERVICE.record_interaction,
                query, response_time, sum(c.score for c in contexts) / len(contexts),
                None, req.user_id, req.conversation_id, ab_config
            )
        
        logger.info(f"Generated enhanced response in {response_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating enhanced response: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit user feedback for learning system."""
    try:
        feedback_id = LEARNING_SYSTEM.collect_feedback(
            req.user_id, req.session_id, req.query, req.response,
            req.citations, req.feedback_type, req.feedback_value, req.metadata
        )
        
        return {
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Feedback collected successfully"
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    conversation = CONVERSATION_MANAGER.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation.conversation_id,
        "turns": [
            {
                "turn_id": turn.turn_id,
                "user_query": turn.user_query,
                "assistant_response": turn.assistant_response,
                "citations": turn.citations,
                "timestamp": turn.timestamp
            }
            for turn in conversation.turns
        ],
        "summary": conversation.get_conversation_summary()
    }

@app.post("/conversations")
async def create_conversation(user_id: Optional[str] = None):
    """Create a new conversation."""
    conversation_id = CONVERSATION_MANAGER.create_conversation(user_id)
    return {"conversation_id": conversation_id}

@app.get("/experiments")
async def list_experiments():
    """List all experiments."""
    return EXPERIMENT_MANAGER.list_experiments()

@app.post("/experiments")
async def create_experiment(req: ExperimentCreateRequest):
    """Create a new A/B test experiment."""
    try:
        experiment_id = EXPERIMENT_MANAGER.create_experiment(
            req.name, req.description, req.variants, req.metrics, req.created_by
        )
        return {"experiment_id": experiment_id, "status": "created"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start an experiment."""
    success = EXPERIMENT_MANAGER.start_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start experiment")
    
    return {"status": "started"}

@app.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop an experiment."""
    success = EXPERIMENT_MANAGER.stop_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to stop experiment")
    
    return {"status": "stopped"}

@app.get("/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Get experiment results."""
    return EXPERIMENT_MANAGER.get_experiment_results(experiment_id)

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary."""
    try:
        from analytics.dashboard import RAGAnalytics
        analytics = RAGAnalytics()
        
        return {
            "performance_metrics": analytics.get_performance_metrics(),
            "usage_trends": analytics.get_usage_trends(),
            "retriever_performance": analytics.get_retriever_performance(),
            "query_analysis": analytics.get_query_analysis(),
            "feedback_analysis": analytics.get_feedback_analysis()
        }
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return {"error": "Analytics not available"}

@app.post("/learning/force_update")
async def force_learning_update():
    """Force learning system update."""
    try:
        stats = LEARNING_SYSTEM.force_learning_update()
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error(f"Learning update error: {e}")
        raise HTTPException(status_code=500, detail="Learning update failed")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return RESPONSE_CACHE.get_cache_stats()

@app.delete("/cache/clear")
async def clear_cache():
    """Clear response cache."""
    success = CACHE_MANAGER.clear()
    return {"status": "success" if success else "failed"}

@app.get("/languages")
async def get_supported_languages():
    """Get supported languages."""
    return MULTILINGUAL_SYSTEM.get_language_stats()

# Create example experiment on startup
@app.on_event("startup")
async def create_example_experiment():
    """Create example experiment if none exist."""
    try:
        experiments = EXPERIMENT_MANAGER.list_experiments()
        if not experiments:
            example_exp = EXAMPLE_EXPERIMENTS['retriever_comparison']
            experiment_id = EXPERIMENT_MANAGER.create_experiment(
                example_exp['name'],
                example_exp['description'],
                example_exp['variants'],
                example_exp['metrics'],
                "system"
            )
            logger.info(f"Created example experiment: {experiment_id}")
    except Exception as e:
        logger.warning(f"Failed to create example experiment: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
