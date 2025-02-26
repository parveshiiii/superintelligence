from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

class EmotionDetection:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=7)

    def detect_emotion(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        return emotions[predictions.item()]