import unittest
from cognitive.forgetting_mechanism import ForgettingMechanism

class TestForgettingMechanism(unittest.TestCase):
    def setUp(self):
        self.forgetting_mechanism = ForgettingMechanism(threshold=0.5)
        self.forgetting_mechanism.add_memory("key1", "value1", relevance=0.3)
        self.forgetting_mechanism.add_memory("key2", "value2", relevance=0.7)

    def test_forget_irrelevant(self):
        self.forgetting_mechanism.forget_irrelevant()
        self.assertIsNone(self.forgetting_mechanism.get_memory("key1"))
        self.assertIsNotNone(self.forgetting_mechanism.get_memory("key2"))

if __name__ == '__main__':
    unittest.main()