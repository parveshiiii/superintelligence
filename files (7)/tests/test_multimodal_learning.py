import unittest
import torch
from multimodal.multimodal_learning import MultiModalLearning

class TestMultiModalLearning(unittest.TestCase):
    def setUp(self):
        self.model = MultiModalLearning()

    def test_image_embeddings(self):
        image = torch.randn(1, 3, 224, 224)
        embeddings = self.model.get_image_embeddings(image)
        self.assertEqual(embeddings.shape, (1, 197, 768))

    def test_audio_embeddings(self):
        audio = torch.randn(1, 16000)
        embeddings = self.model.get_audio_embeddings(audio)
        self.assertEqual(embeddings.shape, (1, 501, 768))

if __name__ == '__main__':
    unittest.main()