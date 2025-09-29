"""
Real-time learning system with feedback-based model improvement.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import pickle

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 scale
    RELEVANCE = "relevance"  # Document-level relevance
    CORRECTION = "correction"  # User provides correct answer
    REPORT = "report"  # Report inappropriate content

@dataclass
class UserFeedback:
    """Represents user feedback on a response."""
    feedback_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    query: str
    response: str
    citations: List[Dict[str, Any]]
    feedback_type: FeedbackType
    feedback_value: Any  # Rating, boolean, text, etc.
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class LearningSignal:
    """Processed learning signal from feedback."""
    signal_id: str
    query: str
    positive_documents: List[str]  # Document IDs that were helpful
    negative_documents: List[str]  # Document IDs that were not helpful
    query_intent: Optional[str]
    confidence: float
    timestamp: str
    source: str  # 'explicit_feedback', 'implicit_behavior', 'correction'

class FeedbackCollector:
    """Collects and processes user feedback."""
    
    def __init__(self, storage_path: str = "store/feedback"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.feedback_history: List[UserFeedback] = []
        self._load_feedback_history()
        
        logger.info(f"FeedbackCollector initialized with {len(self.feedback_history)} historical feedback items")
    
    def collect_feedback(self, user_id: Optional[str], session_id: Optional[str],
                        query: str, response: str, citations: List[Dict[str, Any]],
                        feedback_type: FeedbackType, feedback_value: Any,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Collect user feedback."""
        feedback_id = str(uuid.uuid4())
        
        feedback = UserFeedback(
            feedback_id=feedback_id,
            user_id=user_id,
            session_id=session_id,
            query=query,
            response=response,
            citations=citations,
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.feedback_history.append(feedback)
        self._save_feedback(feedback)
        
        logger.info(f"Collected {feedback_type.value} feedback for query: {query[:50]}...")
        return feedback_id
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        if not self.feedback_history:
            return {"total_feedback": 0}
        
        # Count by type
        type_counts = defaultdict(int)
        for feedback in self.feedback_history:
            type_counts[feedback.feedback_type.value] += 1
        
        # Calculate recent feedback (last 7 days)
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_feedback = [
            f for f in self.feedback_history
            if datetime.fromisoformat(f.timestamp) > recent_cutoff
        ]
        
        # Calculate average ratings
        ratings = [
            f.feedback_value for f in self.feedback_history
            if f.feedback_type == FeedbackType.RATING and isinstance(f.feedback_value, (int, float))
        ]
        
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            "total_feedback": len(self.feedback_history),
            "recent_feedback_7d": len(recent_feedback),
            "feedback_by_type": dict(type_counts),
            "average_rating": round(avg_rating, 2),
            "total_ratings": len(ratings)
        }
    
    def get_query_feedback(self, query: str, similarity_threshold: float = 0.8) -> List[UserFeedback]:
        """Get feedback for similar queries."""
        # Simple similarity based on word overlap
        query_words = set(query.lower().split())
        
        similar_feedback = []
        for feedback in self.feedback_history:
            feedback_words = set(feedback.query.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(query_words & feedback_words)
            union = len(query_words | feedback_words)
            
            if union > 0:
                similarity = intersection / union
                if similarity >= similarity_threshold:
                    similar_feedback.append(feedback)
        
        return similar_feedback
    
    def _save_feedback(self, feedback: UserFeedback):
        """Save feedback to storage."""
        feedback_file = self.storage_path / "feedback.jsonl"
        
        try:
            with open(feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(feedback), ensure_ascii=False, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def _load_feedback_history(self):
        """Load feedback history from storage."""
        feedback_file = self.storage_path / "feedback.jsonl"
        
        if not feedback_file.exists():
            return
        
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        feedback = UserFeedback(
                            feedback_id=data['feedback_id'],
                            user_id=data['user_id'],
                            session_id=data['session_id'],
                            query=data['query'],
                            response=data['response'],
                            citations=data['citations'],
                            feedback_type=FeedbackType(data['feedback_type']),
                            feedback_value=data['feedback_value'],
                            timestamp=data['timestamp'],
                            metadata=data['metadata']
                        )
                        self.feedback_history.append(feedback)
        except Exception as e:
            logger.error(f"Failed to load feedback history: {e}")

class LearningSignalProcessor:
    """Processes feedback into learning signals."""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.learning_signals: List[LearningSignal] = []
        
        logger.info("LearningSignalProcessor initialized")
    
    def process_feedback_batch(self, min_feedback_count: int = 10) -> List[LearningSignal]:
        """Process recent feedback into learning signals."""
        recent_feedback = self._get_recent_feedback()
        
        if len(recent_feedback) < min_feedback_count:
            logger.info(f"Not enough recent feedback ({len(recent_feedback)} < {min_feedback_count})")
            return []
        
        signals = []
        
        # Group feedback by query similarity
        query_groups = self._group_feedback_by_query(recent_feedback)
        
        for query, feedback_group in query_groups.items():
            if len(feedback_group) < 3:  # Need minimum feedback for reliable signal
                continue
            
            signal = self._create_learning_signal_from_group(query, feedback_group)
            if signal:
                signals.append(signal)
        
        self.learning_signals.extend(signals)
        logger.info(f"Generated {len(signals)} learning signals from {len(recent_feedback)} feedback items")
        
        return signals
    
    def _get_recent_feedback(self, days: int = 7) -> List[UserFeedback]:
        """Get recent feedback within specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            f for f in self.feedback_collector.feedback_history
            if datetime.fromisoformat(f.timestamp) > cutoff_date
        ]
    
    def _group_feedback_by_query(self, feedback_list: List[UserFeedback]) -> Dict[str, List[UserFeedback]]:
        """Group feedback by similar queries."""
        query_groups = defaultdict(list)
        
        for feedback in feedback_list:
            # Simple grouping by normalized query
            normalized_query = ' '.join(feedback.query.lower().split())
            query_groups[normalized_query].append(feedback)
        
        return dict(query_groups)
    
    def _create_learning_signal_from_group(self, query: str, feedback_group: List[UserFeedback]) -> Optional[LearningSignal]:
        """Create learning signal from grouped feedback."""
        positive_docs = []
        negative_docs = []
        total_confidence = 0
        
        for feedback in feedback_group:
            confidence = self._calculate_feedback_confidence(feedback)
            total_confidence += confidence
            
            # Extract document relevance
            if self._is_positive_feedback(feedback):
                # Add all cited documents as positive
                for citation in feedback.citations:
                    doc_id = str(citation.get('id', ''))
                    if doc_id and doc_id not in positive_docs:
                        positive_docs.append(doc_id)
            
            elif self._is_negative_feedback(feedback):
                # Add cited documents as negative (they didn't help)
                for citation in feedback.citations:
                    doc_id = str(citation.get('id', ''))
                    if doc_id and doc_id not in negative_docs:
                        negative_docs.append(doc_id)
        
        if not positive_docs and not negative_docs:
            return None
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(feedback_group) if feedback_group else 0
        
        signal = LearningSignal(
            signal_id=str(uuid.uuid4()),
            query=query,
            positive_documents=positive_docs,
            negative_documents=negative_docs,
            query_intent=self._infer_query_intent(query),
            confidence=avg_confidence,
            timestamp=datetime.now().isoformat(),
            source='explicit_feedback'
        )
        
        return signal
    
    def _calculate_feedback_confidence(self, feedback: UserFeedback) -> float:
        """Calculate confidence score for feedback."""
        if feedback.feedback_type == FeedbackType.RATING:
            # Convert 1-5 rating to confidence (1=0.2, 5=1.0)
            return feedback.feedback_value / 5.0 if isinstance(feedback.feedback_value, (int, float)) else 0.5
        
        elif feedback.feedback_type == FeedbackType.THUMBS_UP:
            return 0.8
        
        elif feedback.feedback_type == FeedbackType.THUMBS_DOWN:
            return 0.8
        
        elif feedback.feedback_type == FeedbackType.RELEVANCE:
            return 0.9  # Document-level feedback is usually more reliable
        
        elif feedback.feedback_type == FeedbackType.CORRECTION:
            return 1.0  # User corrections are highly reliable
        
        else:
            return 0.5
    
    def _is_positive_feedback(self, feedback: UserFeedback) -> bool:
        """Check if feedback is positive."""
        if feedback.feedback_type == FeedbackType.THUMBS_UP:
            return True
        
        elif feedback.feedback_type == FeedbackType.RATING:
            return isinstance(feedback.feedback_value, (int, float)) and feedback.feedback_value >= 4
        
        elif feedback.feedback_type == FeedbackType.RELEVANCE:
            return feedback.feedback_value is True
        
        return False
    
    def _is_negative_feedback(self, feedback: UserFeedback) -> bool:
        """Check if feedback is negative."""
        if feedback.feedback_type == FeedbackType.THUMBS_DOWN:
            return True
        
        elif feedback.feedback_type == FeedbackType.RATING:
            return isinstance(feedback.feedback_value, (int, float)) and feedback.feedback_value <= 2
        
        elif feedback.feedback_type == FeedbackType.RELEVANCE:
            return feedback.feedback_value is False
        
        return False
    
    def _infer_query_intent(self, query: str) -> Optional[str]:
        """Infer query intent from text."""
        query_lower = query.lower()
        
        # Simple intent classification
        if any(word in query_lower for word in ['how', 'what', 'why', 'when', 'where']):
            return 'question'
        elif any(word in query_lower for word in ['problem', 'issue', 'broken', 'error']):
            return 'problem'
        elif any(word in query_lower for word in ['cancel', 'refund', 'return']):
            return 'transaction'
        elif any(word in query_lower for word in ['account', 'login', 'password']):
            return 'account'
        else:
            return 'general'

class AdaptiveRetriever:
    """Retriever that adapts based on learning signals."""
    
    def __init__(self, base_retriever, storage_path: str = "store/adaptive_weights"):
        self.base_retriever = base_retriever
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Document weights based on feedback
        self.document_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.query_boosts: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Load existing weights
        self._load_weights()
        
        logger.info("AdaptiveRetriever initialized")
    
    def update_from_signals(self, learning_signals: List[LearningSignal]):
        """Update retriever weights from learning signals."""
        for signal in learning_signals:
            self._apply_learning_signal(signal)
        
        # Save updated weights
        self._save_weights()
        
        logger.info(f"Updated retriever weights from {len(learning_signals)} learning signals")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search with adaptive weights."""
        # Get base results
        results = self.base_retriever.search(query, top_k * 2)  # Get more to allow reweighting
        
        if not results:
            return results
        
        # Apply adaptive weights
        for result in results:
            doc_id = str(result.get('id', ''))
            
            # Apply document-level weight
            doc_weight = self.document_weights.get(doc_id, 1.0)
            
            # Apply query-specific boost
            query_normalized = ' '.join(query.lower().split())
            query_boosts = self.query_boosts.get(query_normalized, {})
            query_boost = query_boosts.get(doc_id, 1.0)
            
            # Combine weights
            final_weight = doc_weight * query_boost
            
            # Update score
            original_score = result.get('score', 0.0)
            result['score'] = original_score * final_weight
            result['adaptive_weight'] = final_weight
        
        # Re-sort by updated scores
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results[:top_k]
    
    def _apply_learning_signal(self, signal: LearningSignal):
        """Apply a learning signal to update weights."""
        learning_rate = 0.1
        confidence_factor = signal.confidence
        
        # Update positive document weights
        for doc_id in signal.positive_documents:
            current_weight = self.document_weights[doc_id]
            # Increase weight for positive documents
            self.document_weights[doc_id] = current_weight + (learning_rate * confidence_factor)
            
            # Update query-specific boost
            query_normalized = ' '.join(signal.query.lower().split())
            current_boost = self.query_boosts[query_normalized].get(doc_id, 1.0)
            self.query_boosts[query_normalized][doc_id] = current_boost + (learning_rate * confidence_factor * 0.5)
        
        # Update negative document weights
        for doc_id in signal.negative_documents:
            current_weight = self.document_weights[doc_id]
            # Decrease weight for negative documents (but keep > 0)
            new_weight = max(0.1, current_weight - (learning_rate * confidence_factor * 0.5))
            self.document_weights[doc_id] = new_weight
            
            # Update query-specific boost (decrease)
            query_normalized = ' '.join(signal.query.lower().split())
            current_boost = self.query_boosts[query_normalized].get(doc_id, 1.0)
            new_boost = max(0.1, current_boost - (learning_rate * confidence_factor * 0.3))
            self.query_boosts[query_normalized][doc_id] = new_boost
    
    def _save_weights(self):
        """Save adaptive weights to storage."""
        weights_file = self.storage_path / "adaptive_weights.pkl"
        
        try:
            weights_data = {
                'document_weights': dict(self.document_weights),
                'query_boosts': dict(self.query_boosts),
                'updated_at': datetime.now().isoformat()
            }
            
            with open(weights_file, 'wb') as f:
                pickle.dump(weights_data, f)
                
        except Exception as e:
            logger.error(f"Failed to save adaptive weights: {e}")
    
    def _load_weights(self):
        """Load adaptive weights from storage."""
        weights_file = self.storage_path / "adaptive_weights.pkl"
        
        if not weights_file.exists():
            return
        
        try:
            with open(weights_file, 'rb') as f:
                weights_data = pickle.load(f)
            
            self.document_weights = defaultdict(lambda: 1.0, weights_data.get('document_weights', {}))
            self.query_boosts = defaultdict(dict, weights_data.get('query_boosts', {}))
            
            logger.info(f"Loaded adaptive weights for {len(self.document_weights)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load adaptive weights: {e}")
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptive weights."""
        if not self.document_weights:
            return {"adapted_documents": 0}
        
        weights = list(self.document_weights.values())
        
        return {
            "adapted_documents": len(self.document_weights),
            "query_specific_boosts": len(self.query_boosts),
            "weight_stats": {
                "min": min(weights),
                "max": max(weights),
                "mean": sum(weights) / len(weights),
                "above_1": sum(1 for w in weights if w > 1.0),
                "below_1": sum(1 for w in weights if w < 1.0)
            }
        }

class RealTimeLearningSystem:
    """Orchestrates the real-time learning system."""
    
    def __init__(self, base_retriever, storage_path: str = "store/learning"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.feedback_collector = FeedbackCollector(str(Path(storage_path) / "feedback"))
        self.signal_processor = LearningSignalProcessor(self.feedback_collector)
        self.adaptive_retriever = AdaptiveRetriever(base_retriever, str(Path(storage_path) / "adaptive"))
        
        # Learning schedule
        self.last_learning_update = None
        self.learning_interval = timedelta(hours=1)  # Update every hour
        
        logger.info("RealTimeLearningSystem initialized")
    
    def collect_feedback(self, user_id: Optional[str], session_id: Optional[str],
                        query: str, response: str, citations: List[Dict[str, Any]],
                        feedback_type: str, feedback_value: Any,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Collect user feedback."""
        try:
            feedback_type_enum = FeedbackType(feedback_type)
        except ValueError:
            logger.error(f"Invalid feedback type: {feedback_type}")
            return ""
        
        return self.feedback_collector.collect_feedback(
            user_id, session_id, query, response, citations,
            feedback_type_enum, feedback_value, metadata
        )
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using adaptive retriever."""
        # Check if learning update is needed
        self._maybe_update_learning()
        
        return self.adaptive_retriever.search(query, top_k)
    
    def force_learning_update(self) -> Dict[str, Any]:
        """Force learning update and return statistics."""
        logger.info("Forcing learning update")
        
        # Process feedback into learning signals
        signals = self.signal_processor.process_feedback_batch(min_feedback_count=5)
        
        # Update adaptive retriever
        self.adaptive_retriever.update_from_signals(signals)
        
        self.last_learning_update = datetime.now()
        
        return {
            "learning_signals_generated": len(signals),
            "feedback_stats": self.feedback_collector.get_feedback_stats(),
            "adaptation_stats": self.adaptive_retriever.get_adaptation_stats(),
            "last_update": self.last_learning_update.isoformat()
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "feedback": self.feedback_collector.get_feedback_stats(),
            "adaptation": self.adaptive_retriever.get_adaptation_stats(),
            "learning_signals": len(self.signal_processor.learning_signals),
            "last_learning_update": self.last_learning_update.isoformat() if self.last_learning_update else None,
            "next_scheduled_update": (
                (self.last_learning_update + self.learning_interval).isoformat()
                if self.last_learning_update else "immediate"
            )
        }
    
    def _maybe_update_learning(self):
        """Update learning if enough time has passed."""
        if (self.last_learning_update is None or 
            datetime.now() - self.last_learning_update > self.learning_interval):
            
            try:
                self.force_learning_update()
            except Exception as e:
                logger.error(f"Failed to update learning: {e}")
