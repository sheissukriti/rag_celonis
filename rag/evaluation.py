"""
Evaluation framework for RAG system performance.
Includes relevance, coherence, faithfulness, and response quality metrics.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Results from evaluating a single query-response pair."""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    
    # Retrieval metrics
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    
    # Answer quality metrics
    relevance_score: float
    coherence_score: float
    faithfulness_score: float
    
    # Overall metrics
    overall_score: float
    response_time: float
    
    # Additional metadata
    num_citations: int
    answer_length: int

class RAGEvaluator:
    """Comprehensive evaluation framework for RAG systems."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the evaluator with embedding model for semantic similarity."""
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"Initialized RAG evaluator with model: {embedding_model}")
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict], 
                          ground_truth_docs: Optional[List[Dict]] = None) -> Tuple[float, float, float]:
        """
        Evaluate retrieval quality using precision, recall, and F1.
        
        Args:
            query: The input query
            retrieved_docs: Documents retrieved by the system
            ground_truth_docs: Known relevant documents (if available)
        
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if not ground_truth_docs:
            # If no ground truth, use semantic similarity as proxy
            return self._evaluate_retrieval_semantic(query, retrieved_docs)
        
        # Standard precision/recall evaluation
        retrieved_ids = set(doc['id'] for doc in retrieved_docs)
        relevant_ids = set(doc['id'] for doc in ground_truth_docs)
        
        if not retrieved_ids:
            return 0.0, 0.0, 0.0
        
        true_positives = len(retrieved_ids & relevant_ids)
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0
        recall = true_positives / len(relevant_ids) if relevant_ids else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def _evaluate_retrieval_semantic(self, query: str, retrieved_docs: List[Dict]) -> Tuple[float, float, float]:
        """Evaluate retrieval using semantic similarity when no ground truth is available."""
        if not retrieved_docs:
            return 0.0, 0.0, 0.0
        
        # Encode query and documents
        query_embedding = self.embedding_model.encode([query])
        doc_texts = [doc['text'] for doc in retrieved_docs]
        doc_embeddings = self.embedding_model.encode(doc_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Use top-k approach: assume top 20% are relevant
        k = max(1, len(retrieved_docs) // 5)
        threshold = np.percentile(similarities, 80) if len(similarities) > 1 else similarities[0]
        
        relevant_count = np.sum(similarities >= threshold)
        precision = relevant_count / len(retrieved_docs)
        recall = min(1.0, relevant_count / k)  # Assume k relevant docs exist
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def evaluate_relevance(self, query: str, answer: str, citations: List[Dict]) -> float:
        """
        Evaluate how relevant the answer is to the query.
        
        Args:
            query: The input query
            answer: Generated answer
            citations: Supporting documents
            
        Returns:
            Relevance score between 0 and 1
        """
        # Semantic similarity between query and answer
        query_embedding = self.embedding_model.encode([query])
        answer_embedding = self.embedding_model.encode([answer])
        
        semantic_similarity = cosine_similarity(query_embedding, answer_embedding)[0][0]
        
        # Check if answer addresses the query intent
        intent_score = self._evaluate_intent_match(query, answer)
        
        # Combine scores
        relevance_score = 0.7 * semantic_similarity + 0.3 * intent_score
        
        return float(np.clip(relevance_score, 0, 1))
    
    def _evaluate_intent_match(self, query: str, answer: str) -> float:
        """Evaluate if the answer matches the query intent using heuristics."""
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Question type detection
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        has_question_word = any(word in query_lower for word in question_words)
        
        # Check for direct answers to questions
        if has_question_word:
            # Look for informative content in answer
            if len(answer.split()) > 3 and not answer.lower().startswith('error'):
                return 0.8
            else:
                return 0.3
        
        # For non-questions, check topical similarity
        query_words = set(query_lower.split())
        answer_words = set(answer_lower.split())
        
        overlap = len(query_words & answer_words)
        total_unique = len(query_words | answer_words)
        
        return overlap / total_unique if total_unique > 0 else 0.0
    
    def evaluate_coherence(self, answer: str) -> float:
        """
        Evaluate the coherence and fluency of the generated answer.
        
        Args:
            answer: Generated answer text
            
        Returns:
            Coherence score between 0 and 1
        """
        if not answer or answer.strip().startswith('Error'):
            return 0.0
        
        # Check for complete sentences
        sentences = re.split(r'[.!?]+', answer.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Basic coherence heuristics
        coherence_score = 1.0
        
        # Penalize very short answers
        if len(answer.split()) < 5:
            coherence_score -= 0.3
        
        # Penalize repetitive content
        words = answer.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio < 0.5:
                coherence_score -= 0.4
        
        # Check for proper sentence structure
        properly_structured = sum(1 for s in sentences if len(s.split()) >= 3)
        structure_score = properly_structured / len(sentences) if sentences else 0
        coherence_score = 0.6 * coherence_score + 0.4 * structure_score
        
        return float(np.clip(coherence_score, 0, 1))
    
    def evaluate_faithfulness(self, answer: str, citations: List[Dict]) -> float:
        """
        Evaluate how faithful the answer is to the source documents.
        
        Args:
            answer: Generated answer
            citations: Source documents used
            
        Returns:
            Faithfulness score between 0 and 1
        """
        if not citations or not answer:
            return 0.0
        
        # Combine all citation texts
        citation_text = " ".join([cite['text'] for cite in citations])
        
        if not citation_text.strip():
            return 0.0
        
        # Semantic similarity between answer and citations
        answer_embedding = self.embedding_model.encode([answer])
        citation_embedding = self.embedding_model.encode([citation_text])
        
        semantic_faithfulness = cosine_similarity(answer_embedding, citation_embedding)[0][0]
        
        # Check for citation references in answer
        citation_refs = len(re.findall(r'\[Doc \d+\]', answer))
        has_citations = citation_refs > 0
        citation_bonus = 0.1 if has_citations else 0.0
        
        # Check for contradictions (basic heuristic)
        contradiction_penalty = self._detect_contradictions(answer, citation_text)
        
        faithfulness_score = semantic_faithfulness + citation_bonus - contradiction_penalty
        
        return float(np.clip(faithfulness_score, 0, 1))
    
    def _detect_contradictions(self, answer: str, citation_text: str) -> float:
        """Detect potential contradictions between answer and citations."""
        # Simple negation detection
        answer_lower = answer.lower()
        citation_lower = citation_text.lower()
        
        negation_words = ['not', 'no', 'never', 'cannot', 'unable', 'impossible']
        
        answer_negations = sum(1 for word in negation_words if word in answer_lower)
        citation_negations = sum(1 for word in negation_words if word in citation_lower)
        
        # If there's a significant difference in negation usage, potential contradiction
        if abs(answer_negations - citation_negations) > 2:
            return 0.2
        
        return 0.0
    
    def evaluate_response(self, query: str, answer: str, citations: List[Dict], 
                         response_time: float, ground_truth_docs: Optional[List[Dict]] = None) -> EvaluationResult:
        """
        Comprehensive evaluation of a single query-response pair.
        
        Args:
            query: Input query
            answer: Generated answer
            citations: Retrieved documents used
            response_time: Time taken to generate response
            ground_truth_docs: Known relevant documents (optional)
            
        Returns:
            EvaluationResult with all metrics
        """
        # Retrieval evaluation
        precision, recall, f1 = self.evaluate_retrieval(query, citations, ground_truth_docs)
        
        # Answer quality evaluation
        relevance = self.evaluate_relevance(query, answer, citations)
        coherence = self.evaluate_coherence(answer)
        faithfulness = self.evaluate_faithfulness(answer, citations)
        
        # Overall score (weighted combination)
        overall_score = (
            0.3 * f1 +           # Retrieval quality
            0.3 * relevance +    # Answer relevance
            0.2 * coherence +    # Answer coherence
            0.2 * faithfulness   # Answer faithfulness
        )
        
        return EvaluationResult(
            query=query,
            answer=answer,
            citations=citations,
            retrieval_precision=precision,
            retrieval_recall=recall,
            retrieval_f1=f1,
            relevance_score=relevance,
            coherence_score=coherence,
            faithfulness_score=faithfulness,
            overall_score=overall_score,
            response_time=response_time,
            num_citations=len(citations),
            answer_length=len(answer.split())
        )
    
    def evaluate_batch(self, evaluation_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate multiple query-response pairs and return aggregate statistics.
        
        Args:
            evaluation_data: List of dicts with 'query', 'answer', 'citations', 'response_time'
            
        Returns:
            Dictionary with aggregate evaluation metrics
        """
        if not evaluation_data:
            return {}
        
        results = []
        for item in evaluation_data:
            result = self.evaluate_response(
                query=item['query'],
                answer=item['answer'],
                citations=item.get('citations', []),
                response_time=item.get('response_time', 0.0),
                ground_truth_docs=item.get('ground_truth_docs')
            )
            results.append(result)
        
        # Calculate aggregate statistics
        metrics = {
            'num_evaluations': len(results),
            'avg_retrieval_precision': np.mean([r.retrieval_precision for r in results]),
            'avg_retrieval_recall': np.mean([r.retrieval_recall for r in results]),
            'avg_retrieval_f1': np.mean([r.retrieval_f1 for r in results]),
            'avg_relevance_score': np.mean([r.relevance_score for r in results]),
            'avg_coherence_score': np.mean([r.coherence_score for r in results]),
            'avg_faithfulness_score': np.mean([r.faithfulness_score for r in results]),
            'avg_overall_score': np.mean([r.overall_score for r in results]),
            'avg_response_time': np.mean([r.response_time for r in results]),
            'avg_num_citations': np.mean([r.num_citations for r in results]),
            'avg_answer_length': np.mean([r.answer_length for r in results])
        }
        
        # Add standard deviations
        for metric in ['overall_score', 'relevance_score', 'coherence_score', 'faithfulness_score']:
            values = [getattr(r, metric) for r in results]
            metrics[f'std_{metric}'] = np.std(values)
        
        return metrics
    
    def save_evaluation_results(self, results: List[EvaluationResult], output_path: str):
        """Save evaluation results to JSON file."""
        output_data = []
        for result in results:
            output_data.append({
                'query': result.query,
                'answer': result.answer,
                'num_citations': result.num_citations,
                'retrieval_precision': result.retrieval_precision,
                'retrieval_recall': result.retrieval_recall,
                'retrieval_f1': result.retrieval_f1,
                'relevance_score': result.relevance_score,
                'coherence_score': result.coherence_score,
                'faithfulness_score': result.faithfulness_score,
                'overall_score': result.overall_score,
                'response_time': result.response_time,
                'answer_length': result.answer_length
            })
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} evaluation results to {output_path}")

def load_test_queries(file_path: str) -> List[Dict]:
    """Load test queries from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_test_queries() -> List[Dict]:
    """Create a set of test queries for evaluation."""
    return [
        {
            "query": "I ordered a laptop but it arrived with a broken screen. What should I do?",
            "expected_topics": ["returns", "damaged goods", "replacement"]
        },
        {
            "query": "I need help resetting my password",
            "expected_topics": ["password reset", "account access"]
        },
        {
            "query": "My cat chewed my phone charger. Is this covered under warranty?",
            "expected_topics": ["warranty", "accidental damage"]
        },
        {
            "query": "How long does shipping usually take?",
            "expected_topics": ["shipping", "delivery time"]
        },
        {
            "query": "Can I cancel my order?",
            "expected_topics": ["order cancellation", "refund"]
        },
        {
            "query": "The product I received is different from what I ordered",
            "expected_topics": ["wrong item", "returns", "exchange"]
        },
        {
            "query": "How do I track my order?",
            "expected_topics": ["order tracking", "shipping status"]
        },
        {
            "query": "I want to return an item but I lost the receipt",
            "expected_topics": ["returns", "no receipt", "proof of purchase"]
        },
        {
            "query": "Is there a student discount available?",
            "expected_topics": ["discounts", "student pricing"]
        },
        {
            "query": "The website is not loading properly",
            "expected_topics": ["technical issues", "website problems"]
        }
    ]

if __name__ == "__main__":
    # Example usage
    evaluator = RAGEvaluator()
    test_queries = create_test_queries()
    
    # Save test queries for future use
    with open("evaluation/test_queries.json", "w") as f:
        json.dump(test_queries, f, indent=2)
    
    print(f"Created {len(test_queries)} test queries for evaluation")
