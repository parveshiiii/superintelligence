import unittest
from emotional.intelligence import EmotionDetection

class TestEmotionDetection(unittest.TestCase):
    def setUp(self):
        self.model = EmotionDetection()

    def test_detect_emotion(self):
        text = "I am so happy today!"
        emotion = self.model.detect_emotion(text)
        self.assertEqual(emotion, "joy")

if __name__ == '__main__':
    unittest.main()