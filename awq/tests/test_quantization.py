import unittest
from unittest.mock import patch, MagicMock
from app.quantization import run_quantization, validate_quantized_model

class TestQuantization(unittest.TestCase):
    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    def test_run_quantization(self, mock_tokenizer, mock_awq):
        mock_model = MagicMock()
        mock_awq.from_pretrained.return_value = mock_model
        
        run_quantization('/path/to/model', {'w_bit': 4}, '/path/to/output')
        
        mock_awq.from_pretrained.assert_called_once()
        mock_model.quantize.assert_called_once()
        mock_model.save_quantized.assert_called_once()

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    def test_validate_quantized_model(self, mock_tokenizer, mock_awq):
        mock_model = MagicMock()
        mock_awq.from_quantized.return_value = mock_model
        mock_model.generate.return_value = MagicMock()
        
        result = validate_quantized_model('/path/to/quantized_model')
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()