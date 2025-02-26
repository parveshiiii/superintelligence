from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

@dataclass
class MemoryUnit:
    content: Any
    timestamp: datetime
    tags: List[str]
    importance: float
    context: Dict[str, Any]
    associations: List[str]
    
@dataclass
class MemoryBlock:
    units: List[MemoryUnit]
    category: str
    retention_score: float
    last_accessed: datetime
    
class MemoryMetrics:
    def __init__(self):
        self.access_count: int = 0
        self.last_retrieval: Optional[datetime] = None
        self.importance_score: float = 0.0
        self.relevance_score: float = 0.0