"""
Multi-turn conversation management with context-aware dialogue handling.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    turn_id: str
    user_query: str
    assistant_response: str
    citations: List[Dict[str, Any]]
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class ConversationContext:
    """Maintains context across multiple turns."""
    conversation_id: str
    user_id: Optional[str]
    turns: List[ConversationTurn]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    
    def add_turn(self, user_query: str, assistant_response: str, 
                 citations: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        """Add a new turn to the conversation."""
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            user_query=user_query,
            assistant_response=assistant_response,
            citations=citations,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.turns.append(turn)
        self.updated_at = datetime.now().isoformat()
    
    def get_recent_context(self, max_turns: int = 3, max_length: int = 1000) -> str:
        """Get recent conversation context for prompt enhancement."""
        if not self.turns:
            return ""
        
        recent_turns = self.turns[-max_turns:]
        context_parts = []
        current_length = 0
        
        for turn in recent_turns:
            turn_text = f"User: {turn.user_query}\nAssistant: {turn.assistant_response}"
            if current_length + len(turn_text) > max_length:
                break
            context_parts.append(turn_text)
            current_length += len(turn_text)
        
        return "\n\n".join(context_parts)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        if not self.turns:
            return {"total_turns": 0, "topics": [], "duration": 0}
        
        # Calculate duration
        start_time = datetime.fromisoformat(self.created_at)
        end_time = datetime.fromisoformat(self.updated_at)
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Extract topics (simplified approach)
        topics = []
        for turn in self.turns:
            # Simple keyword extraction for topics
            query_words = turn.user_query.lower().split()
            topics.extend([word for word in query_words if len(word) > 4])
        
        # Get unique topics
        unique_topics = list(set(topics))[:5]  # Top 5 topics
        
        return {
            "total_turns": len(self.turns),
            "topics": unique_topics,
            "duration_minutes": round(duration_minutes, 2),
            "last_activity": self.updated_at
        }

class ConversationManager:
    """Manages multi-turn conversations with persistence."""
    
    def __init__(self, storage_path: str = "store/conversations", 
                 max_conversation_age_days: int = 30):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.max_age_days = max_conversation_age_days
        self.active_conversations: Dict[str, ConversationContext] = {}
        
        # Load active conversations from storage
        self._load_conversations()
        
        logger.info(f"ConversationManager initialized with {len(self.active_conversations)} active conversations")
    
    def create_conversation(self, user_id: Optional[str] = None) -> str:
        """Create a new conversation and return its ID."""
        conversation_id = str(uuid.uuid4())
        
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            turns=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata={}
        )
        
        self.active_conversations[conversation_id] = context
        self._save_conversation(context)
        
        logger.info(f"Created new conversation: {conversation_id}")
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get a conversation by ID."""
        return self.active_conversations.get(conversation_id)
    
    def add_turn(self, conversation_id: str, user_query: str, 
                 assistant_response: str, citations: List[Dict[str, Any]], 
                 metadata: Dict[str, Any] = None) -> bool:
        """Add a turn to an existing conversation."""
        if conversation_id not in self.active_conversations:
            logger.warning(f"Conversation not found: {conversation_id}")
            return False
        
        context = self.active_conversations[conversation_id]
        context.add_turn(user_query, assistant_response, citations, metadata)
        
        # Save updated conversation
        self._save_conversation(context)
        
        logger.info(f"Added turn to conversation {conversation_id}")
        return True
    
    def get_conversation_context(self, conversation_id: str, 
                               max_turns: int = 3, max_length: int = 1000) -> str:
        """Get conversation context for prompt enhancement."""
        context = self.get_conversation(conversation_id)
        if not context:
            return ""
        
        return context.get_recent_context(max_turns, max_length)
    
    def list_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """List all conversations for a user."""
        user_conversations = []
        
        for context in self.active_conversations.values():
            if context.user_id == user_id:
                summary = context.get_conversation_summary()
                summary['conversation_id'] = context.conversation_id
                summary['created_at'] = context.created_at
                user_conversations.append(summary)
        
        # Sort by last activity
        user_conversations.sort(key=lambda x: x['last_activity'], reverse=True)
        return user_conversations
    
    def cleanup_old_conversations(self):
        """Remove conversations older than max_age_days."""
        cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
        conversations_to_remove = []
        
        for conv_id, context in self.active_conversations.items():
            last_update = datetime.fromisoformat(context.updated_at)
            if last_update < cutoff_date:
                conversations_to_remove.append(conv_id)
        
        for conv_id in conversations_to_remove:
            self._archive_conversation(conv_id)
            del self.active_conversations[conv_id]
        
        if conversations_to_remove:
            logger.info(f"Cleaned up {len(conversations_to_remove)} old conversations")
    
    def _save_conversation(self, context: ConversationContext):
        """Save conversation to storage."""
        file_path = self.storage_path / f"{context.conversation_id}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(context), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save conversation {context.conversation_id}: {e}")
    
    def _load_conversations(self):
        """Load conversations from storage."""
        if not self.storage_path.exists():
            return
        
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert dict back to ConversationContext
                turns = [ConversationTurn(**turn) for turn in data['turns']]
                context = ConversationContext(
                    conversation_id=data['conversation_id'],
                    user_id=data['user_id'],
                    turns=turns,
                    created_at=data['created_at'],
                    updated_at=data['updated_at'],
                    metadata=data['metadata']
                )
                
                self.active_conversations[context.conversation_id] = context
                
            except Exception as e:
                logger.error(f"Failed to load conversation from {file_path}: {e}")
    
    def _archive_conversation(self, conversation_id: str):
        """Archive an old conversation."""
        archive_dir = self.storage_path / "archived"
        archive_dir.mkdir(exist_ok=True)
        
        source_file = self.storage_path / f"{conversation_id}.json"
        archive_file = archive_dir / f"{conversation_id}.json"
        
        try:
            if source_file.exists():
                source_file.rename(archive_file)
        except Exception as e:
            logger.error(f"Failed to archive conversation {conversation_id}: {e}")

class ContextAwarePromptBuilder:
    """Enhanced prompt builder that incorporates conversation context."""
    
    def __init__(self, conversation_manager: ConversationManager):
        self.conversation_manager = conversation_manager
    
    def build_context_aware_prompt(self, query: str, contexts: List[Dict[str, Any]], 
                                 conversation_id: Optional[str] = None,
                                 template_type: str = "default") -> str:
        """Build a prompt that includes conversation history."""
        
        # Get conversation context if available
        conversation_context = ""
        if conversation_id:
            conversation_context = self.conversation_manager.get_conversation_context(
                conversation_id, max_turns=3, max_length=800
            )
        
        # Build context from retrieved documents
        doc_context = self._build_document_context(contexts)
        
        # Choose appropriate template based on conversation state
        if conversation_context:
            return self._build_multi_turn_prompt(query, doc_context, conversation_context)
        else:
            return self._build_single_turn_prompt(query, doc_context, template_type)
    
    def _build_document_context(self, contexts: List[Dict[str, Any]], max_length: int = 1500) -> str:
        """Build context from retrieved documents."""
        if not contexts:
            return "No relevant context found."
        
        cited = []
        current_length = 0
        
        for i, doc in enumerate(contexts, start=1):
            doc_text = doc.get('text', '').strip()
            if len(doc_text) > 400:
                doc_text = doc_text[:397] + "..."
            
            citation = f"[Doc {i}] (Score: {doc.get('score', 0):.3f})\n{doc_text}\n"
            
            if current_length + len(citation) > max_length and cited:
                break
            
            cited.append(citation)
            current_length += len(citation)
        
        return "\n".join(cited)
    
    def _build_multi_turn_prompt(self, query: str, doc_context: str, 
                               conversation_context: str) -> str:
        """Build prompt for multi-turn conversations."""
        instructions = (
            "You are a helpful customer support assistant engaged in an ongoing conversation. "
            "Use the conversation history to understand context and provide relevant responses. "
            "Reference previous exchanges when appropriate and maintain conversation continuity."
        )
        
        prompt = (
            f"System: {instructions}\n\n"
            f"Conversation History:\n{conversation_context}\n\n"
            f"Relevant Knowledge Base Information:\n{doc_context}\n\n"
            f"Current Customer Query: {query}\n\n"
            f"Support Agent Response:"
        )
        
        return prompt
    
    def _build_single_turn_prompt(self, query: str, doc_context: str, 
                                template_type: str = "default") -> str:
        """Build prompt for single-turn conversations."""
        instructions = (
            "You are a helpful customer support assistant. Provide accurate, "
            "helpful responses based on the provided context. Be empathetic and professional."
        )
        
        prompt = (
            f"System: {instructions}\n\n"
            f"Context from knowledge base:\n{doc_context}\n\n"
            f"Customer Query: {query}\n\n"
            f"Support Agent Response:"
        )
        
        return prompt
