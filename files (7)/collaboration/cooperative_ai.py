from typing import List, Any

class CooperativeAI:
    def __init__(self):
        self.collaboration_log = []

    def collaborate(self, task: str, partners: List[Any]) -> str:
        collaboration_result = f"Collaborated on {task} with {partners}"
        self.collaboration_log.append(collaboration_result)
        return collaboration_result

    def get_collaboration_log(self) -> List[str]:
        return self.collaboration_log