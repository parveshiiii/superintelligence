import unittest
import torch
from cognitive.deeper_thought import DeeperThoughtProcess, ThoughtProcessConfig

class TestDeeperThoughtProcess(unittest.TestCase):
    def setUp(self):
        config = ThoughtProcessConfig(
            depth=5,
            learning_rate=0.001,
            batch_size=16
        )
        self.deeper_thought_process = DeeperThoughtProcess(config)

    def test_forward(self):
        x = torch.randn(10, 100)
        output = self.deeper_thought_process(x)
        self.assertEqual(output.shape, (10, 100))

if __name__ == '__main__':
    unittest.main()