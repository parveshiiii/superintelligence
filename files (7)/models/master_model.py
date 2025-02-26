from transformers import ViTModel, ViTFeatureExtractor, Wav2Vec2Model, Wav2Vec2Processor
import torch
import torch.nn as nn

class MasterModel(nn.Module):
    def __init__(self):
        super(MasterModel, self).__init__()
        self.image_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.image_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.audio_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        self.audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        self.fc = nn.Linear(768 * 2, 768)  # Assuming both models output 768-dim features

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        inputs = self.image_processor(images=image, return_tensors='pt')
        outputs = self.image_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def forward_audio(self, audio: torch.Tensor) -> torch.Tensor:
        inputs = self.audio_processor(audio, return_tensors='pt')
        outputs = self.audio_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        image_features = self.forward_image(image)
        audio_features = self.forward_audio(audio)
        combined_features = torch.cat((image_features, audio_features), dim=1)
        output = self.fc(combined_features)
        return output