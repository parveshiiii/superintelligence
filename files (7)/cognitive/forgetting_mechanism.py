from typing import Any, Dict, List
import numpy as np

class ForgettingMechanism:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.memory: Dict[str, Any] = {}
        self.relevance_scores: Dict[str, float] = {}

    def add_memory(self, key: str, value: Any, relevance: float) -> None:
        self.memory[key] = value
        self.relevance_scores[key] = relevance

    def update_relevance(self, key: str, relevance: float) -> None:
        if key in self.relevance_scores:
            self.relevance_scores[key] = relevance

    def forget_irrelevant(self) -> None:
        keys_to_forget = [key for key, score in self.relevance_scores.items() if score < self.threshold]
        for key in keys_to_forget:
            del self.memory[key]
            del self.relevance_scores[key]

    def get_memory(self, key: str) -> Any:
        return self.memory.get(key, None)

    def __repr__(self) -> str:
        return f"Memory: {self.memory}, Relevance Scores: {self.relevance_scores}"