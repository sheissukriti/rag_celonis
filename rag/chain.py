
from typing import List
from dataclasses import dataclass


@dataclass
class ContextDoc:
    id: int
    score: float
    text: str


def build_prompt(query: str, contexts: List[ContextDoc]) -> str:
    cited = []
    for i, c in enumerate(contexts, start=1):
        cited.append(f"[Doc {i}]\n{c.text}\n")
    context_block = "\n".join(cited)
    instructions = (
        "You are a helpful customer support assistant. Use only the provided context. "
        "Cite sources inline as [Doc i]. If the answer is not in context, ask a "
        "clarifying question or state that you need more details. Be concise and actionable."
    )
    prompt = (
        f"System: {instructions}\n\n"
        f"Context:\n{context_block}\n"
        f"User: {query}\n"
        f"Assistant:"
    )
    return prompt
