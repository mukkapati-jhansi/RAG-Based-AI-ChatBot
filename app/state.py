from typing import TypedDict, List, Dict


class RAGState(TypedDict):
    question: str
    context: List[Dict]
    answer: str
    confidence: float