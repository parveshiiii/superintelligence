from typing import Dict, List
import importlib
import sys
import logging
from pathlib import Path

class ConnectionValidator:
    def __init__(self):
        self.required_modules = [
            'safety.core.value_alignment',
            'safety.controls.evolution_control',
            'cognitive.meta_learning',
            'evolution.controllers.evolution_controller',
            'resources.manager',
            'monitoring.system_monitor'
        ]
        
        self.module_dependencies = {
            'safety.core.value_alignment': ['numpy', 'torch'],
            'cognitive.meta_learning': ['torch', 'numpy'],
            'evolution.controllers.evolution_controller': ['numpy'],
            'resources.manager': ['psutil', 'torch'],
            'monitoring.system_monitor': ['psutil', 'logging']
        }
        
    def validate_all_connections(self) -> Dict[str, bool]:
        """Validate all module connections"""
        results = {}
        
        # Check module imports
        for module in self.required_modules:
            try:
                importlib.import_module(module)
                results[module] = True
            except ImportError as e:
                logging.error(f"Failed to import {module}: {str(e)}")
                results[module] = False
                
        # Check dependencies
        for module, deps in self.module_dependencies.items():
            for dep in deps:
                if dep not in sys.modules:
                    results[f"{module}_dep_{dep}"] = False
                    logging.error(f"Missing dependency {dep} for {module}")
                    
        return results
        
    def check_file_structure(self) -> Dict[str, bool]:
        """Verify file structure"""
        required_paths = [
            'safety/core',
            'safety/controls',
            'cognitive',
            'evolution/controllers',
            'resources',
            'monitoring'
        ]
        
        results = {}
        for path in required_paths:
            full_path = Path(path)
            results[path] = full_path.exists()
            if not full_path.exists():
                logging.error(f"Missing directory: {path}")
                
        return results