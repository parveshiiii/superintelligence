from typing import Dict, Tuple
import psutil
import torch
from dataclasses import dataclass

@dataclass
class ResourceLimits:
    max_memory_percent: float
    max_cpu_percent: float
    max_gpu_memory_percent: float
    max_disk_usage_percent: float

class ResourceManager:
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.resource_history: List[Dict[str, float]] = []
        
    def check_resources(self) -> Tuple[bool, Dict[str, float]]:
        """Check current resource usage against limits"""
        current_usage = {
            "memory": psutil.virtual_memory().percent / 100,
            "cpu": psutil.cpu_percent() / 100,
            "disk": psutil.disk_usage('/').percent / 100,
            "gpu": self._get_gpu_usage()
        }
        
        self.resource_history.append(current_usage)
        
        is_safe = all(
            usage <= limit for usage, limit in zip(
                current_usage.values(),
                [self.limits.max_memory_percent,
                 self.limits.max_cpu_percent,
                 self.limits.max_disk_usage_percent,
                 self.limits.max_gpu_memory_percent]
            )
        )
        
        return is_safe, current_usage
        
    def _get_gpu_usage(self) -> float:
        # Implementation of GPU usage retrieval
        pass