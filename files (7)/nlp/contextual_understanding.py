from transformers import BertTokenizer, BertModel
import torch

class ContextualUnderstanding:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def get_contextual_embeddings(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state