from transformers import ViTModel, ViTFeatureExtractor
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch

class MultiModalLearning:
    def __init__(self):
        self.image_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.image_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.audio_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        self.audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

    def get_image_embeddings(self, image: torch.Tensor) -> torch.Tensor:
        inputs = self.image_processor(images=image, return_tensors='pt')
        outputs = self.image_model(**inputs)
        return outputs.last_hidden_state

    def get_audio_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        inputs = self.audio_processor(audio, return_tensors='pt')
        outputs = self.audio_model(**inputs)
        return outputs.last_hidden_state