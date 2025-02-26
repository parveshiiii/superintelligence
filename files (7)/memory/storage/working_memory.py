from typing import Dict, List, Optional
from ..core.memory_types import MemoryUnit, MemoryMetrics
import numpy as np

class WorkingMemory:
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[MemoryUnit] = []
        self.metrics: Dict[str, MemoryMetrics] = {}
        
    def add_item(self, item: MemoryUnit) -> bool:
        if len(self.items) >= self.capacity:
            self._consolidate_memory()
        self.items.append(item)
        self.metrics[str(id(item))] = MemoryMetrics()
        return True
        
    def _consolidate_memory(self) -> None:
        # Implementation of memory consolidation logic
        pass