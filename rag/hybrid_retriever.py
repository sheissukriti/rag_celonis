"""
Hybrid retrieval system combining multiple retrieval strategies.
Implements ensemble methods for improved retrieval performance.
"""

import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import logging
from .retrieve import BM25Retriever, FaissTfidfRetriever, DenseRetriever

logger = logging.getLogger(__name__)

@dataclass
class RetrievalStrategy:
    """Configuration for a single retrieval strategy."""
    name: str
    retriever: Union[BM25Retriever, FaissTfidfRetriever, DenseRetriever]
    weight: float
    
class HybridRetriever:
    """
    Hybrid retrieval system that combines multiple retrieval strategies.
    
    Supports:
    - BM25 (lexical matching)
    - TF-IDF + SVD (dense-ish semantic matching)
    - Dense embeddings (semantic matching)
    - Weighted combination and rank fusion
    """
    
    def __init__(self, strategies: List[RetrievalStrategy]):
        """
        Initialize hybrid retriever with multiple strategies.
        
        Args:
            strategies: List of RetrievalStrategy objects
        """
        self.strategies = strategies
        self.total_weight = sum(s.weight for s in strategies)
        logger.info(f"Initialized hybrid retriever with {len(strategies)} strategies")
    
    def search(self, query: str, top_k: int = 10, fusion_method: str = "weighted_score") -> List[Dict]:
        """
        Search using hybrid retrieval with multiple strategies.
        
        Args:
            query: Search query
            top_k: Number of results to return
            fusion_method: Method to combine results ("weighted_score", "rrf", "max_score")
            
        Returns:
            List of retrieved documents with combined scores
        """
        if not self.strategies:
            return []
        
        # Collect results from all strategies
        all_results = {}  # doc_id -> {strategy_name: score, text: str}
        
        for strategy in self.strategies:
            try:
                results = strategy.retriever.search(query, top_k=top_k * 2)  # Get more for fusion
                logger.debug(f"Strategy {strategy.name} retrieved {len(results)} documents")
                
                for doc in results:
                    doc_id = doc['id']
                    if doc_id not in all_results:
                        all_results[doc_id] = {'text': doc['text'], 'scores': {}}
                    
                    all_results[doc_id]['scores'][strategy.name] = doc['score']
                    
            except Exception as e:
                logger.error(f"Error in strategy {strategy.name}: {e}")
                continue
        
        if not all_results:
            logger.warning("No results from any retrieval strategy")
            return []
        
        # Apply fusion method
        if fusion_method == "weighted_score":
            final_results = self._weighted_score_fusion(all_results)
        elif fusion_method == "rrf":
            final_results = self._reciprocal_rank_fusion(all_results)
        elif fusion_method == "max_score":
            final_results = self._max_score_fusion(all_results)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Sort by final score and return top_k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:top_k]
    
    def _weighted_score_fusion(self, all_results: Dict) -> List[Dict]:
        """Combine scores using weighted average."""
        final_results = []
        
        for doc_id, doc_data in all_results.items():
            scores = doc_data['scores']
            
            # Calculate weighted score
            weighted_sum = 0.0
            total_weight = 0.0
            
            for strategy in self.strategies:
                if strategy.name in scores:
                    weighted_sum += scores[strategy.name] * strategy.weight
                    total_weight += strategy.weight
            
            if total_weight > 0:
                final_score = weighted_sum / total_weight
                final_results.append({
                    'id': doc_id,
                    'score': final_score,
                    'text': doc_data['text'],
                    'strategy_scores': scores
                })
        
        return final_results
    
    def _reciprocal_rank_fusion(self, all_results: Dict, k: int = 60) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank)) for each strategy where doc appears
        """
        # First, get rankings from each strategy
        strategy_rankings = {}
        
        for strategy in self.strategies:
            # Sort documents by score for this strategy
            docs_with_scores = []
            for doc_id, doc_data in all_results.items():
                if strategy.name in doc_data['scores']:
                    docs_with_scores.append((doc_id, doc_data['scores'][strategy.name]))
            
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            strategy_rankings[strategy.name] = {doc_id: rank for rank, (doc_id, _) in enumerate(docs_with_scores)}
        
        # Calculate RRF scores
        final_results = []
        for doc_id, doc_data in all_results.items():
            rrf_score = 0.0
            
            for strategy in self.strategies:
                if strategy.name in strategy_rankings and doc_id in strategy_rankings[strategy.name]:
                    rank = strategy_rankings[strategy.name][doc_id]
                    rrf_score += strategy.weight * (1.0 / (k + rank))
            
            if rrf_score > 0:
                final_results.append({
                    'id': doc_id,
                    'score': rrf_score,
                    'text': doc_data['text'],
                    'strategy_scores': doc_data['scores']
                })
        
        return final_results
    
    def _max_score_fusion(self, all_results: Dict) -> List[Dict]:
        """Take the maximum score across all strategies."""
        final_results = []
        
        for doc_id, doc_data in all_results.items():
            scores = doc_data['scores']
            
            if scores:
                max_score = max(scores.values())
                final_results.append({
                    'id': doc_id,
                    'score': max_score,
                    'text': doc_data['text'],
                    'strategy_scores': scores
                })
        
        return final_results

def create_hybrid_retriever(config: Dict) -> HybridRetriever:
    """
    Create a hybrid retriever based on configuration.
    
    Args:
        config: Configuration dictionary with retriever settings
        
    Returns:
        Configured HybridRetriever instance
    """
    strategies = []
    
    # Add BM25 strategy if enabled
    if config.get('bm25', {}).get('enabled', False):
        from .retrieve import BM25Retriever
        bm25_retriever = BM25Retriever(config['bm25']['qa_dump_path'])
        strategies.append(RetrievalStrategy(
            name="bm25",
            retriever=bm25_retriever,
            weight=config['bm25'].get('weight', 0.4)
        ))
    
    # Add TF-IDF strategy if enabled
    if config.get('tfidf', {}).get('enabled', False):
        from .retrieve import FaissTfidfRetriever
        tfidf_retriever = FaissTfidfRetriever(
            index_path=config['tfidf']['index_path'],
            ids_path=config['tfidf']['ids_path'],
            qa_dump_path=config['tfidf']['qa_dump_path']
        )
        strategies.append(RetrievalStrategy(
            name="tfidf",
            retriever=tfidf_retriever,
            weight=config['tfidf'].get('weight', 0.6)
        ))
    
    # Add dense strategy if enabled
    if config.get('dense', {}).get('enabled', False):
        from .retrieve import DenseRetriever
        dense_retriever = DenseRetriever(
            embedding_model=config['dense']['embedding_model'],
            embeddings_path=config['dense']['embeddings_path'],
            ids_path=config['dense']['ids_path'],
            qa_dump_path=config['dense']['qa_dump_path']
        )
        strategies.append(RetrievalStrategy(
            name="dense",
            retriever=dense_retriever,
            weight=config['dense'].get('weight', 0.5)
        ))
    
    if not strategies:
        raise ValueError("No retrieval strategies enabled in configuration")
    
    return HybridRetriever(strategies)

# Example configuration for hybrid retrieval
HYBRID_CONFIG_EXAMPLE = {
    "bm25": {
        "enabled": True,
        "weight": 0.3,
        "qa_dump_path": "store/qa_texts_10k.jsonl"
    },
    "tfidf": {
        "enabled": True,
        "weight": 0.7,
        "index_path": "store/faiss.index",
        "ids_path": "store/faiss.ids",
        "qa_dump_path": "store/qa_texts_10k.jsonl"
    },
    "dense": {
        "enabled": False,
        "weight": 0.5,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embeddings_path": "store/faiss.emb.npy",
        "ids_path": "store/faiss.ids",
        "qa_dump_path": "store/qa_texts_10k.jsonl"
    }
}
