import unittest
from unittest.mock import patch, MagicMock
import os
from app.quantization import run_quantization
from app.config import Config

class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.model_path = "/path/to/model"
        self.quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM"
        }
        self.output_dir = "/output/dir"

    @patch('app.quantization.AutoAWQForCausalLM.from_pretrained')
    @patch('app.quantization.AutoTokenizer.from_pretrained')
    def test_run_quantization_success(self, mock_tokenizer, mock_from_pretrained):
        # Setup
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model
        mock_tokenizer.return_value = MagicMock()

        # Run the function
        run_quantization(self.model_path, self.quant_config, self.output_dir)

        # Assertions
        mock_from_pretrained.assert_called_once_with(self.model_path, low_cpu_mem_usage=True, use_cache=False)
        mock_tokenizer.assert_called_once_with(self.model_path, trust_remote_code=True)
        mock_model.quantize.assert_called_once_with(mock_tokenizer.return_value, quant_config=self.quant_config)
        mock_model.save_quantized.assert_called_once_with(self.output_dir)

    @patch('app.quantization.AutoAWQForCausalLM.from_pretrained')
    def test_run_quantization_failure(self, mock_from_pretrained):
        # Setup
        mock_from_pretrained.side_effect = Exception("Failed to load model")

        # Run the function and check for exception
        with self.assertRaises(Exception):
            run_quantization(self.model_path, self.quant_config, self.output_dir)

if __name__ == '__main__':
    unittest.main()
