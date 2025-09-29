"""
Advanced reranking using cross-encoder models for better relevance scoring.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None

logger = logging.getLogger(__name__)

@dataclass
class RerankResult:
    """Result of reranking operation."""
    id: Union[int, str]
    text: str
    original_score: float
    rerank_score: float
    final_score: float
    metadata: Dict[str, Any] = None

class BaseReranker(ABC):
    """Base class for all rerankers."""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to query."""
        pass
    
    @abstractmethod
    def get_relevance_scores(self, query: str, documents: List[str]) -> List[float]:
        """Get relevance scores for query-document pairs."""
        pass

class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranker for high-quality relevance scoring."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_length: int = 512, batch_size: int = 32):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            max_length: Maximum sequence length
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None
        
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("CrossEncoder not available. Install sentence-transformers.")
            return
        
        try:
            self.model = CrossEncoder(model_name, max_length=max_length)
            logger.info(f"Loaded CrossEncoder model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            self.model = None
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder model."""
        if not self.model:
            logger.warning("CrossEncoder model not available, returning original ranking")
            return documents[:top_k] if top_k else documents
        
        if not documents:
            return documents
        
        try:
            # Prepare query-document pairs
            texts = [doc.get('text', '') for doc in documents]
            pairs = [(query, text) for text in texts]
            
            # Get relevance scores in batches
            scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                batch_scores = self.model.predict(batch)
                scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else batch_scores)
            
            # Create reranked results
            reranked = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                reranked_doc = doc.copy()
                reranked_doc['original_score'] = doc.get('score', 0.0)
                reranked_doc['rerank_score'] = float(score)
                reranked_doc['score'] = float(score)  # Use rerank score as final score
                reranked.append(reranked_doc)
            
            # Sort by rerank score
            reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"Reranked {len(documents)} documents using CrossEncoder")
            return reranked[:top_k] if top_k else reranked
            
        except Exception as e:
            logger.error(f"Error in CrossEncoder reranking: {e}")
            return documents[:top_k] if top_k else documents
    
    def get_relevance_scores(self, query: str, documents: List[str]) -> List[float]:
        """Get relevance scores for query-document pairs."""
        if not self.model:
            return [0.0] * len(documents)
        
        try:
            pairs = [(query, doc) for doc in documents]
            scores = []
            
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                batch_scores = self.model.predict(batch)
                scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else batch_scores)
            
            return [float(score) for score in scores]
            
        except Exception as e:
            logger.error(f"Error getting relevance scores: {e}")
            return [0.0] * len(documents)

class HybridReranker(BaseReranker):
    """Hybrid reranker combining multiple reranking strategies."""
    
    def __init__(self, rerankers: List[BaseReranker], weights: Optional[List[float]] = None):
        """
        Initialize hybrid reranker.
        
        Args:
            rerankers: List of reranker instances
            weights: Weights for combining scores (defaults to equal weights)
        """
        self.rerankers = rerankers
        self.weights = weights or [1.0 / len(rerankers)] * len(rerankers)
        
        if len(self.weights) != len(rerankers):
            raise ValueError("Number of weights must match number of rerankers")
        
        logger.info(f"Initialized HybridReranker with {len(rerankers)} rerankers")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank using weighted combination of multiple rerankers."""
        if not documents:
            return documents
        
        # Get scores from all rerankers
        all_scores = []
        for reranker in self.rerankers:
            try:
                texts = [doc.get('text', '') for doc in documents]
                scores = reranker.get_relevance_scores(query, texts)
                all_scores.append(scores)
            except Exception as e:
                logger.error(f"Error in reranker {type(reranker).__name__}: {e}")
                all_scores.append([0.0] * len(documents))
        
        # Normalize scores to [0, 1] range
        normalized_scores = []
        for scores in all_scores:
            if max(scores) > min(scores):
                min_score, max_score = min(scores), max(scores)
                normalized = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized = [0.5] * len(scores)  # All scores are the same
            normalized_scores.append(normalized)
        
        # Combine scores with weights
        final_scores = []
        for i in range(len(documents)):
            weighted_score = sum(
                weight * scores[i] 
                for weight, scores in zip(self.weights, normalized_scores)
            )
            final_scores.append(weighted_score)
        
        # Create reranked results
        reranked = []
        for i, (doc, score) in enumerate(zip(documents, final_scores)):
            reranked_doc = doc.copy()
            reranked_doc['original_score'] = doc.get('score', 0.0)
            reranked_doc['rerank_score'] = float(score)
            reranked_doc['score'] = float(score)
            reranked.append(reranked_doc)
        
        # Sort by final score
        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Hybrid reranked {len(documents)} documents")
        return reranked[:top_k] if top_k else reranked
    
    def get_relevance_scores(self, query: str, documents: List[str]) -> List[float]:
        """Get combined relevance scores."""
        if not documents:
            return []
        
        # Get scores from all rerankers
        all_scores = []
        for reranker in self.rerankers:
            try:
                scores = reranker.get_relevance_scores(query, documents)
                all_scores.append(scores)
            except Exception as e:
                logger.error(f"Error in reranker {type(reranker).__name__}: {e}")
                all_scores.append([0.0] * len(documents))
        
        # Normalize and combine
        normalized_scores = []
        for scores in all_scores:
            if max(scores) > min(scores):
                min_score, max_score = min(scores), max(scores)
                normalized = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized = [0.5] * len(scores)
            normalized_scores.append(normalized)
        
        # Combine with weights
        final_scores = []
        for i in range(len(documents)):
            weighted_score = sum(
                weight * scores[i] 
                for weight, scores in zip(self.weights, normalized_scores)
            )
            final_scores.append(weighted_score)
        
        return final_scores

class SemanticSimilarityReranker(BaseReranker):
    """Reranker based on semantic similarity using sentence transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize semantic similarity reranker."""
        self.model_name = model_name
        self.model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded SentenceTransformer model: {model_name}")
        except ImportError:
            logger.warning("SentenceTransformer not available")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank documents using semantic similarity."""
        if not self.model or not documents:
            return documents[:top_k] if top_k else documents
        
        try:
            texts = [doc.get('text', '') for doc in documents]
            
            # Encode query and documents
            query_embedding = self.model.encode([query])
            doc_embeddings = self.model.encode(texts)
            
            # Calculate cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Create reranked results
            reranked = []
            for i, (doc, sim_score) in enumerate(zip(documents, similarities)):
                reranked_doc = doc.copy()
                reranked_doc['original_score'] = doc.get('score', 0.0)
                reranked_doc['rerank_score'] = float(sim_score)
                reranked_doc['score'] = float(sim_score)
                reranked.append(reranked_doc)
            
            # Sort by similarity score
            reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"Semantic reranked {len(documents)} documents")
            return reranked[:top_k] if top_k else reranked
            
        except Exception as e:
            logger.error(f"Error in semantic reranking: {e}")
            return documents[:top_k] if top_k else documents
    
    def get_relevance_scores(self, query: str, documents: List[str]) -> List[float]:
        """Get semantic similarity scores."""
        if not self.model or not documents:
            return [0.0] * len(documents)
        
        try:
            query_embedding = self.model.encode([query])
            doc_embeddings = self.model.encode(documents)
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            return [float(sim) for sim in similarities]
            
        except Exception as e:
            logger.error(f"Error getting semantic similarity scores: {e}")
            return [0.0] * len(documents)

def create_advanced_reranker(config: Dict[str, Any]) -> BaseReranker:
    """Factory function to create reranker based on configuration."""
    reranker_type = config.get('type', 'cross_encoder')
    
    if reranker_type == 'cross_encoder':
        model_name = config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        return CrossEncoderReranker(model_name=model_name)
    
    elif reranker_type == 'semantic':
        model_name = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        return SemanticSimilarityReranker(model_name=model_name)
    
    elif reranker_type == 'hybrid':
        # Create multiple rerankers for hybrid approach
        rerankers = []
        
        # Add cross-encoder if available
        if CROSS_ENCODER_AVAILABLE:
            rerankers.append(CrossEncoderReranker())
        
        # Add semantic similarity reranker
        rerankers.append(SemanticSimilarityReranker())
        
        weights = config.get('weights', None)
        return HybridReranker(rerankers, weights)
    
    else:
        logger.warning(f"Unknown reranker type: {reranker_type}, using CrossEncoder")
        return CrossEncoderReranker()

# Example configuration
RERANKER_CONFIGS = {
    'cross_encoder': {
        'type': 'cross_encoder',
        'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    },
    'semantic': {
        'type': 'semantic',
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
    },
    'hybrid': {
        'type': 'hybrid',
        'weights': [0.7, 0.3]  # Higher weight for cross-encoder
    }
}
