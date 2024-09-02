import unittest
from unittest.mock import patch, MagicMock
import torch
from app.quantization import run_quantization, validate_quantized_model, validate_quant_config, get_quantized_model_size

class TestQuantization(unittest.TestCase):
    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    def test_run_quantization(self, mock_memory_allocated, mock_get_device_properties, mock_cuda_available, mock_tokenizer, mock_awq):
        mock_cuda_available.return_value = True
        mock_get_device_properties.return_value.total_memory = 16 * 1024 * 1024 * 1024  # 16GB
        mock_memory_allocated.return_value = 0
        
        mock_model = MagicMock()
        mock_awq.from_pretrained.return_value = mock_model
        
        quant_config = {'w_bit': 4, 'q_group_size': 128, 'zero_point': True, 'version': 'GEMM'}
        run_quantization('/path/to/model', quant_config, '/path/to/output')
        
        mock_awq.from_pretrained.assert_called_once()
        mock_model.quantize.assert_called_once_with(mock_tokenizer.from_pretrained.return_value, quant_config=quant_config)
        mock_model.save_quantized.assert_called_once_with('/path/to/output')

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    @patch('torch.cuda.is_available')
    def test_run_quantization_cpu(self, mock_cuda_available, mock_tokenizer, mock_awq):
        mock_cuda_available.return_value = False
        
        mock_model = MagicMock()
        mock_awq.from_pretrained.return_value = mock_model
        
        quant_config = {'w_bit': 4, 'q_group_size': 128, 'zero_point': True, 'version': 'GEMM'}
        run_quantization('/path/to/model', quant_config, '/path/to/output')
        
        mock_awq.from_pretrained.assert_called_once_with(
            '/path/to/model',
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            device_map=None
        )

    @patch('app.quantization.AutoAWQForCausalLM')
    def test_run_quantization_cuda_oom(self, mock_awq):
        mock_awq.from_pretrained.side_effect = RuntimeError("CUDA out of memory")
        
        with self.assertRaises(RuntimeError) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("CUDA out of memory", str(context.exception))

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    def test_validate_quantized_model(self, mock_tokenizer, mock_awq):
        mock_model = MagicMock()
        mock_awq.from_quantized.return_value = mock_model
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value.decode.return_value = "Generated text"
        
        result = validate_quantized_model('/path/to/quantized_model')
        self.assertTrue(result)

    @patch('app.quantization.AutoAWQForCausalLM')
    def test_validate_quantized_model_failure(self, mock_awq):
        mock_awq.from_quantized.side_effect = Exception("Model loading failed")
        
        result = validate_quantized_model('/path/to/quantized_model')
        self.assertFalse(result)

    def test_validate_quant_config(self):
        valid_config = {
            'zero_point': True,
            'q_group_size': 128,
            'w_bit': 4,
            'version': 'GEMM'
        }
        validate_quant_config(valid_config)  # Should not raise an exception

        invalid_configs = [
            {'zero_point': 'True', 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'},
            {'zero_point': True, 'q_group_size': -1, 'w_bit': 4, 'version': 'GEMM'},
            {'zero_point': True, 'q_group_size': 128, 'w_bit': 3, 'version': 'GEMM'},
            {'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'INVALID'},
            {'zero_point': True, 'q_group_size': 128, 'w_bit': 16, 'version': 'GEMM'},
            {'zero_point': True, 'q_group_size': 0, 'w_bit': 4, 'version': 'GEMM'},
        ]
        for config in invalid_configs:
            with self.assertRaises(ValueError):
                validate_quant_config(config)

    @patch('os.walk')
    @patch('os.path.getsize')
    def test_get_quantized_model_size(self, mock_getsize, mock_walk):
        mock_walk.return_value = [
            ('/path/to/model', [], ['file1.bin', 'file2.bin']),
            ('/path/to/model/subdir', [], ['file3.bin'])
        ]
        mock_getsize.side_effect = [1024 * 1024, 2 * 1024 * 1024, 3 * 1024 * 1024]
        
        result = get_quantized_model_size('/path/to/model')
        self.assertEqual(result, 6 * 1024 * 1024)

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    def test_run_quantization_attribute_error(self, mock_tokenizer, mock_awq):
        mock_model = MagicMock()
        mock_awq.from_pretrained.return_value = mock_model
        mock_tokenizer.from_pretrained.side_effect = AttributeError("'AutoTokenizer' object has no attribute 'quantize'")

        with self.assertRaises(AttributeError):
            run_quantization('/path/to/model', {}, '/path/to/output')

    @patch('app.quantization.AutoAWQForCausalLM')
    def test_run_quantization_runtime_error(self, mock_awq):
        mock_awq.from_pretrained.side_effect = RuntimeError("Cannot copy out of meta tensor")
        
        with self.assertRaises(RuntimeError) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("Meta device error", str(context.exception))

    @patch('app.quantization.AutoAWQForCausalLM')
    def test_run_quantization_device_mismatch(self, mock_awq):
        mock_awq.from_pretrained.side_effect = RuntimeError("Expected all tensors to be on the same device")
        
        with self.assertRaises(RuntimeError) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("Device mismatch error", str(context.exception))

    @patch('app.quantization.AutoAWQForCausalLM')
    def test_run_quantization_model_offloading(self, mock_awq):
        mock_awq.from_pretrained.side_effect = RuntimeError("You can't move a model that has some modules offloaded to cpu or disk")
        
        with self.assertRaises(RuntimeError) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("Model offloading error", str(context.exception))

    def test_validate_quant_config_missing_keys(self):
        invalid_config = {'w_bit': 4}  # Missing other required keys
        with self.assertRaises(ValueError):
            validate_quant_config(invalid_config)

    def test_validate_quant_config_invalid_version(self):
        invalid_config = {
            'zero_point': True,
            'q_group_size': 128,
            'w_bit': 4,
            'version': 'INVALID'
        }
        with self.assertRaises(ValueError):
            validate_quant_config(invalid_config)

if __name__ == '__main__':
    unittest.main()