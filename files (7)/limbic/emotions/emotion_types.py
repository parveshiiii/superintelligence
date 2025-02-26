from dataclasses import dataclass
from typing import Dict, List

@dataclass
class EmotionState:
    happiness: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    disgust: float = 0.0
    surprise: float = 0.0
    
    def dominant_emotion(self) -> str:
        emotions = {
            "happiness": self.happiness,
            "sadness": self.sadness,
            "anger": self.anger,
            "fear": self.fear,
            "disgust": self.disgust,
            "surprise": self.surprise
        }
        return max(emotions.items(), key=lambda x: x[1])[0]