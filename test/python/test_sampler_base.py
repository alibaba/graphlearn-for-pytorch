import unittest

import torch
from torch.testing import assert_close

from graphlearn_torch.sampler.base import NodeSamplerInput


def _assert_tensor_equal(tensor1, tensor2):
    assert_close(tensor1, tensor2, rtol=0, atol=0)


class TestSamplerBase(unittest.TestCase):

    def test_node_sampler_input_int_index(self):
        input_data = NodeSamplerInput(node=torch.arange(10))
        _assert_tensor_equal(input_data[0].node, torch.tensor([0]))

    def test_node_sampler_input_tensor_index(self):
        input_data = NodeSamplerInput(node=torch.arange(10))
        with self.subTest("scalar tensor input"):
            _assert_tensor_equal(input_data[torch.tensor(0)].node, torch.tensor([0]))

        with self.subTest("slice tensor input"):
            _assert_tensor_equal(input_data[torch.tensor([0, 1])].node, torch.tensor([0, 1]))

    def test_node_sampler_input_multiple_examples(self):
        input_data = NodeSamplerInput(node=torch.tensor([[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(len(input_data), 2)

        with self.subTest("scalar tensor input"):
            _assert_tensor_equal(input_data[torch.tensor(0)].node, torch.tensor([0, 1, 2]))

        with self.subTest("slice tensor input"):
            _assert_tensor_equal(
                input_data[torch.tensor([0, 1])].node, torch.tensor([0, 1, 2, 3, 4, 5])
            )

        # Also test with dataloader - since that's how we actually use this
        with self.subTest("dataloader"):
            loader = torch.utils.data.DataLoader(torch.arange(2))
            expected_data = torch.tensor([[0, 1, 2], [3, 4, 5]])

            self.assertEqual(len(loader), len(expected_data))
            for data, expected in zip(loader, expected_data):
                _assert_tensor_equal(input_data[data].node, expected)

        # Also test with dataloader - since that's how we actually use this
        with self.subTest("dataloader - batched"):
            loader = torch.utils.data.DataLoader(torch.arange(2), batch_size=2)
            expected_data = torch.tensor([[0, 1, 2, 3, 4, 5]])

            self.assertEqual(len(loader), len(expected_data))
            for data, expected in zip(loader, expected_data):
                _assert_tensor_equal(input_data[data].node, expected)


if __name__ == "__main__":
    unittest.main()
