
import unittest
import torch
from epochor.generators.synthetic_v1 import SyntheticBenchmarkerV1

class TestSyntheticDataGenerator(unittest.TestCase):

    def test_generator_shape_and_dtype(self):
        length = 100
        n_series = 5
        benchmarker = SyntheticBenchmarkerV1(length=length, n_series=n_series)
        data = benchmarker.prepare_data(seed=123)

        # Assertions for inputs_padded
        self.assertIn("inputs_padded", data)
        inputs_padded = data["inputs_padded"]
        self.assertIsInstance(inputs_padded, torch.Tensor)
        self.assertEqual(inputs_padded.dtype, torch.float32)
        self.assertEqual(inputs_padded.dim(), 2)
        self.assertEqual(inputs_padded.shape[0], n_series)
        # The exact padded length is data-dependent, so we check it's within a reasonable range
        self.assertGreater(inputs_padded.shape[1], 0)
        self.assertLessEqual(inputs_padded.shape[1], length)

        # Assertions for attention_mask
        self.assertIn("attention_mask", data)
        attention_mask = data["attention_mask"]
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(attention_mask.dtype, torch.long)
        self.assertEqual(attention_mask.shape, inputs_padded.shape)

        # Assertions for targets_padded
        self.assertIn("targets_padded", data)
        targets_padded = data["targets_padded"]
        self.assertIsInstance(targets_padded, torch.Tensor)
        self.assertEqual(targets_padded.dtype, torch.float32)
        self.assertEqual(targets_padded.dim(), 2)
        self.assertEqual(targets_padded.shape[0], n_series)
        self.assertGreater(targets_padded.shape[1], 0)
        self.assertLessEqual(targets_padded.shape[1], length)

        # Assertions for actual_target_lengths
        self.assertIn("actual_target_lengths", data)
        actual_target_lengths = data["actual_target_lengths"]
        self.assertIsInstance(actual_target_lengths, list)
        self.assertEqual(len(actual_target_lengths), n_series)
        self.assertTrue(all(isinstance(l, int) for l in actual_target_lengths))

    def test_determinism(self):
        benchmarker = SyntheticBenchmarkerV1(length=50, n_series=3)
        
        # Generate data with a fixed seed
        data1 = benchmarker.prepare_data(seed=42)
        inputs1 = data1["inputs_padded"]
        targets1 = data1["targets_padded"]

        # Generate data again with the same seed
        data2 = benchmarker.prepare_data(seed=42)
        inputs2 = data2["inputs_padded"]
        targets2 = data2["targets_padded"]

        # The generated data should be identical
        self.assertTrue(torch.equal(inputs1, inputs2))
        self.assertTrue(torch.equal(targets1, targets2))

    def test_edge_cases(self):
        # Zero seed
        try:
            benchmarker = SyntheticBenchmarkerV1(length=20, n_series=2)
            benchmarker.prepare_data(seed=0)
        except Exception as e:
            self.fail(f"Generator failed with seed=0: {e}")

        # Invalid constructor arguments
        with self.assertRaises(ValueError):
            SyntheticBenchmarkerV1(length=1, n_series=10) # length <= 1
        with self.assertRaises(ValueError):
            SyntheticBenchmarkerV1(length=10, n_series=0) # n_series <= 0
        with self.assertRaises(ValueError):
            SyntheticBenchmarkerV1(length=10, n_series=10, min_input_frac=0, max_input_frac=0.9) # min_input_frac not > 0
        with self.assertRaises(ValueError):
            SyntheticBenchmarkerV1(length=10, n_series=10, min_input_frac=0.5, max_input_frac=1.0) # max_input_frac not < 1
        with self.assertRaises(ValueError):
            SyntheticBenchmarkerV1(length=10, n_series=10, min_input_frac=0.8, max_input_frac=0.7) # min >= max

if __name__ == '__main__':
    unittest.main()
