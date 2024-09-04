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
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization(self, mock_getsize, mock_listdir, mock_memory_allocated, mock_get_device_properties, mock_cuda_available, mock_tokenizer, mock_awq):
        mock_cuda_available.return_value = True
        mock_get_device_properties.return_value.total_memory = 16 * 1024 * 1024 * 1024  # 16GB
        mock_memory_allocated.return_value = 0
        mock_listdir.return_value = ['model.bin']
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        
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
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_cpu(self, mock_getsize, mock_listdir, mock_cuda_available, mock_tokenizer, mock_awq):
        mock_cuda_available.return_value = False
        mock_listdir.return_value = ['model.bin']
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        
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
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_cuda_oom(self, mock_getsize, mock_listdir, mock_awq):
        mock_listdir.return_value = ['model.bin']
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        mock_awq.from_pretrained.side_effect = RuntimeError("CUDA out of memory")
        
        with self.assertRaises(RuntimeError) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("CUDA out of memory", str(context.exception))

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_validate_quantized_model(self, mock_getsize, mock_listdir, mock_tokenizer, mock_awq):
        mock_model = MagicMock()
        mock_awq.from_quantized.return_value = mock_model
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value.decode.return_value = "Generated text"
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        mock_listdir.return_value = ['model.bin']
        
        result = validate_quantized_model('/path/to/quantized_model')
        self.assertTrue(result)

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_validate_quantized_model_failure(self, mock_getsize, mock_listdir, mock_awq):
        mock_awq.from_quantized.side_effect = Exception("Model loading failed")
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        mock_listdir.return_value = ['model.bin']
        
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
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_attribute_error(self, mock_getsize, mock_listdir, mock_tokenizer, mock_awq):
        mock_listdir.return_value = ['model.bin']
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        mock_model = MagicMock()
        mock_awq.from_pretrained.return_value = mock_model
        mock_tokenizer.from_pretrained.side_effect = AttributeError("'AutoTokenizer' object has no attribute 'quantize'")

        with self.assertRaises(AttributeError):
            run_quantization('/path/to/model', {}, '/path/to/output')

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_attribute_error_general(self, mock_getsize, mock_listdir, mock_tokenizer, mock_awq):
        mock_listdir.return_value = ['model.bin']
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        mock_model = MagicMock()
        mock_awq.from_pretrained.return_value = mock_model
        mock_model.quantize.side_effect = AttributeError("General attribute error")

        with self.assertRaises(AttributeError) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("General attribute error", str(context.exception))

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_runtime_error(self, mock_getsize, mock_listdir, mock_awq):
        mock_listdir.return_value = ['model.bin']
        mock_awq.from_pretrained.side_effect = RuntimeError("Cannot copy out of meta tensor")
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        
        with self.assertRaises(RuntimeError) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("Meta device error", str(context.exception))

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_device_mismatch(self, mock_getsize, mock_listdir, mock_awq):
        mock_listdir.return_value = ['model.bin']
        mock_awq.from_pretrained.side_effect = RuntimeError("Expected all tensors to be on the same device")
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        
        with self.assertRaises(RuntimeError) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("Device mismatch error", str(context.exception))

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_model_offloading(self, mock_getsize, mock_listdir, mock_awq):
        mock_listdir.return_value = ['model.bin']
        mock_awq.from_pretrained.side_effect = RuntimeError("You can't move a model that has some modules offloaded to cpu or disk")
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        
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

    def test_validate_quant_config_missing_keys(self):
        invalid_config = {}  # Empty config
        with self.assertRaises(ValueError):
            validate_quant_config(invalid_config)

    def test_validate_quant_config_invalid_types(self):
        invalid_config = {
            'zero_point': 1,  # Should be boolean
            'q_group_size': '128',  # Should be integer
            'w_bit': '4',  # Should be integer
            'version': 1  # Should be string
        }
        with self.assertRaises(ValueError):
            validate_quant_config(invalid_config)

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_general_exception(self, mock_getsize, mock_listdir, mock_tokenizer, mock_awq):
        mock_listdir.return_value = ['model.bin']
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        mock_awq.from_pretrained.side_effect = Exception("General error")
        
        with self.assertRaises(Exception) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("General error", str(context.exception))

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_insufficient_gpu_memory(self, mock_getsize, mock_listdir, mock_memory_allocated, mock_get_device_properties, mock_cuda_available, mock_tokenizer, mock_awq):
        mock_cuda_available.return_value = True
        mock_get_device_properties.return_value.total_memory = 4 * 1024 * 1024 * 1024  # 4GB
        mock_memory_allocated.return_value = 3 * 1024 * 1024 * 1024  # 3GB used
        mock_listdir.return_value = ['model.bin']
        mock_getsize.return_value = 2 * 1024 * 1024 * 1024  # 2GB model size

        with self.assertLogs(level='WARNING') as log:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertTrue(any("Insufficient GPU memory" in message for message in log.output))
    
    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_invalid_config(self, mock_getsize, mock_listdir, mock_tokenizer, mock_awq):
        mock_listdir.return_value = ['model.bin']
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        invalid_config = {'w_bit': 3, 'q_group_size': 128, 'zero_point': True, 'version': 'GEMM'}
        with self.assertRaises(ValueError) as context:
            run_quantization('/path/to/model', invalid_config, '/path/to/output')
        self.assertIn("Invalid quant_config", str(context.exception))

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    @patch('torch.cuda.is_available')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_no_model_files(self, mock_getsize, mock_listdir, mock_cuda_available, mock_tokenizer, mock_awq):
        mock_cuda_available.return_value = True
        mock_listdir.return_value = []
        
        with self.assertRaises(ValueError) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("No model files found", str(context.exception))

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    @patch('torch.cuda.is_available')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_unsupported_model(self, mock_getsize, mock_listdir, mock_cuda_available, mock_tokenizer, mock_awq):
        mock_cuda_available.return_value = True
        mock_listdir.return_value = ['model.bin']
        mock_getsize.return_value = 1024 * 1024 * 1024  # 1GB
        mock_awq.from_pretrained.side_effect = ValueError("Unsupported model type")
        
        with self.assertRaises(ValueError) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("Unsupported model type", str(context.exception))

    def test_get_quantized_model_size_empty_directory(self):
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = []
        result = get_quantized_model_size('/path/to/empty/model')
        self.assertEqual(result, 0)

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_run_quantization_quantize_error(self, mock_getsize, mock_listdir, mock_tokenizer, mock_awq):
        mock_listdir.return_value = ['model.bin']
        mock_getsize.return_value = 1024.0 * 1024 * 1024  # 1GB as a float
        mock_model = MagicMock()
        mock_awq.from_pretrained.return_value = mock_model
        mock_model.quantize.side_effect = RuntimeError("Quantization error")

        with self.assertRaises(RuntimeError) as context:
            run_quantization('/path/to/model', {}, '/path/to/output')
        
        self.assertIn("Quantization error", str(context.exception))

    @patch('app.quantization.AutoAWQForCausalLM')
    @patch('app.quantization.AutoTokenizer')
    def test_validate_quantized_model_generation_error(self, mock_tokenizer, mock_awq):
        mock_model = MagicMock()
        mock_awq.from_quantized.return_value = mock_model
        mock_model.generate.side_effect = RuntimeError("Generation error")
        
        result = validate_quantized_model('/path/to/quantized_model')
        self.assertFalse(result)

    def test_get_quantized_model_size_access_error(self):
        with patch('os.walk') as mock_walk, patch('os.path.getsize') as mock_getsize:
            mock_walk.return_value = [('/path/to/model', [], ['file1.bin'])]
            mock_getsize.side_effect = OSError("Access denied")
            result = get_quantized_model_size('/path/to/model')
            self.assertEqual(result, 0)

if __name__ == '__main__':
    unittest.main()