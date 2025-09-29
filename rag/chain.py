
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)

@dataclass
class ContextDoc:
    id: int
    score: float
    text: str

class PromptBuilder:
    """Enhanced prompt builder with multiple templates and strategies."""
    
    def __init__(self, template_type: str = "default"):
        self.template_type = template_type
        self.templates = {
            "default": self._default_template,
            "conversational": self._conversational_template,
            "structured": self._structured_template,
            "concise": self._concise_template
        }
    
    def build_prompt(self, query: str, contexts: List[ContextDoc], 
                    max_context_length: int = 2000) -> str:
        """
        Build a prompt with the specified template.
        
        Args:
            query: User query
            contexts: Retrieved context documents
            max_context_length: Maximum length of context section
            
        Returns:
            Formatted prompt string
        """
        if self.template_type not in self.templates:
            logger.warning(f"Unknown template type: {self.template_type}, using default")
            template_func = self.templates["default"]
        else:
            template_func = self.templates[self.template_type]
        
        return template_func(query, contexts, max_context_length)
    
    def _default_template(self, query: str, contexts: List[ContextDoc], max_context_length: int) -> str:
        """Default template with clear instructions and citations."""
        # Build context with length limits
        context_block = self._build_context_block(contexts, max_context_length)
        
        instructions = (
            "You are a helpful customer support assistant. Your goal is to provide accurate, "
            "helpful responses based on the provided context from previous customer interactions.\n\n"
            "Guidelines:\n"
            "- Use ONLY the information provided in the context\n"
            "- Cite sources using [Doc X] format when referencing information\n"
            "- Be empathetic and professional in your tone\n"
            "- Provide actionable steps when possible\n"
            "- If the context doesn't contain enough information, ask clarifying questions\n"
            "- Keep responses concise but comprehensive"
        )
        
        prompt = (
            f"System: {instructions}\n\n"
            f"Context from previous customer support interactions:\n{context_block}\n\n"
            f"Customer Query: {query}\n\n"
            f"Support Agent Response:"
        )
        
        return prompt
    
    def _conversational_template(self, query: str, contexts: List[ContextDoc], max_context_length: int) -> str:
        """Conversational template for more natural responses."""
        context_block = self._build_context_block(contexts, max_context_length)
        
        instructions = (
            "You are a friendly customer support agent helping a customer. "
            "Use the provided examples to understand how to respond appropriately. "
            "Be warm, helpful, and solution-oriented. Always cite your sources with [Doc X]."
        )
        
        prompt = (
            f"{instructions}\n\n"
            f"Here are some examples of how similar issues were handled:\n{context_block}\n\n"
            f"Now, please help this customer:\n\"{query}\"\n\n"
            f"Your response:"
        )
        
        return prompt
    
    def _structured_template(self, query: str, contexts: List[ContextDoc], max_context_length: int) -> str:
        """Structured template for systematic responses."""
        context_block = self._build_context_block(contexts, max_context_length)
        
        instructions = (
            "You are a customer support specialist. Provide a structured response with:\n"
            "1. Acknowledgment of the issue\n"
            "2. Relevant information from the context (with [Doc X] citations)\n"
            "3. Clear next steps or recommendations\n"
            "4. Additional help offer if needed"
        )
        
        prompt = (
            f"Instructions: {instructions}\n\n"
            f"Reference Material:\n{context_block}\n\n"
            f"Customer Issue: {query}\n\n"
            f"Structured Response:\n"
        )
        
        return prompt
    
    def _concise_template(self, query: str, contexts: List[ContextDoc], max_context_length: int) -> str:
        """Concise template for brief, direct responses."""
        context_block = self._build_context_block(contexts, max_context_length, max_docs=3)
        
        instructions = (
            "Provide a brief, direct answer to the customer's question. "
            "Use the context provided and cite sources with [Doc X]. "
            "Be helpful but concise."
        )
        
        prompt = (
            f"{instructions}\n\n"
            f"Context: {context_block}\n\n"
            f"Q: {query}\n"
            f"A:"
        )
        
        return prompt
    
    def _build_context_block(self, contexts: List[ContextDoc], max_length: int, max_docs: Optional[int] = None) -> str:
        """
        Build context block with length and document limits.
        
        Args:
            contexts: List of context documents
            max_length: Maximum total length of context
            max_docs: Maximum number of documents to include
            
        Returns:
            Formatted context block
        """
        if not contexts:
            return "No relevant context found."
        
        # Limit number of documents if specified
        if max_docs:
            contexts = contexts[:max_docs]
        
        cited = []
        current_length = 0
        
        for i, doc in enumerate(contexts, start=1):
            # Clean and truncate document text
            doc_text = self._clean_text(doc.text)
            
            # Format citation
            citation = f"[Doc {i}] (Score: {doc.score:.3f})\n{doc_text}\n"
            
            # Check length limit
            if current_length + len(citation) > max_length and cited:
                break
            
            cited.append(citation)
            current_length += len(citation)
        
        return "\n".join(cited) if cited else "No relevant context available."
    
    def _clean_text(self, text: str) -> str:
        """Clean and format document text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Truncate very long documents
        if len(text) > 500:
            text = text[:497] + "..."
        
        return text

# Backward compatibility function
def build_prompt(query: str, contexts: List[ContextDoc], template_type: str = "default") -> str:
    """
    Build a prompt using the specified template.
    
    Args:
        query: User query
        contexts: Retrieved context documents
        template_type: Template to use (default, conversational, structured, concise)
        
    Returns:
        Formatted prompt string
    """
    builder = PromptBuilder(template_type)
    return builder.build_prompt(query, contexts)

def analyze_query_intent(query: str) -> Dict[str, any]:
    """
    Analyze query to determine intent and appropriate response strategy.
    
    Args:
        query: User query text
        
    Returns:
        Dictionary with intent analysis results
    """
    query_lower = query.lower()
    
    # Intent categories
    intents = {
        "question": any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']),
        "problem": any(word in query_lower for word in ['problem', 'issue', 'broken', 'error', 'not working', 'failed']),
        "request": any(word in query_lower for word in ['need', 'want', 'help', 'can you', 'please']),
        "complaint": any(word in query_lower for word in ['angry', 'frustrated', 'terrible', 'awful', 'worst']),
        "urgent": any(word in query_lower for word in ['urgent', 'asap', 'immediately', 'emergency'])
    }
    
    # Determine primary intent
    primary_intent = max(intents.items(), key=lambda x: x[1])
    
    # Suggest appropriate template
    template_suggestions = {
        "question": "structured",
        "problem": "conversational", 
        "request": "default",
        "complaint": "conversational",
        "urgent": "concise"
    }
    
    suggested_template = template_suggestions.get(primary_intent[0], "default")
    
    return {
        "intents": intents,
        "primary_intent": primary_intent[0],
        "suggested_template": suggested_template,
        "confidence": primary_intent[1]
    }
