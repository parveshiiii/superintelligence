from datetime import datetime

class SystemStatus:
    def __init__(self, is_safe, resources_available, evolution_allowed, current_stage, active_components, timestamp):
        self.is_safe = is_safe
        self.resources_available = resources_available
        self.evolution_allowed = evolution_allowed
        self.current_stage = current_stage
        self.active_components = active_components
        self.timestamp = timestamp