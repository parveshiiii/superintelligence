from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class IntegrationConfig:
    """Configuration for model integration"""
    model_paths: Dict[str, str]
    gpu_enabled: bool
    memory_limit: int
    optimization_level: str
    logging_level: str
    auto_improvement: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_paths': self.model_paths,
            'gpu_enabled': self.gpu_enabled,
            'memory_limit': self.memory_limit,
            'optimization_level': self.optimization_level,
            'logging_level': self.logging_level,
            'auto_improvement': self.auto_improvement
        }