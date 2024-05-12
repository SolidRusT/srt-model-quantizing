import unittest
from app.converter import convert_model_to_safetensors
from unittest.mock import patch, mock_open

class TestConverter(unittest.TestCase):
    @patch('app.converter.glob.glob')
    @patch('app.converter.torch.load')
    @patch('app.converter.save_file')
    def test_convert_model_to_safetensors(self, mock_save_file, mock_torch_load, mock_glob):
        # Setup
        mock_glob.return_value = ['model.bin']
        mock_torch_load.return_value = {'weights': 'tensor_data'}

        # Test action
        result = convert_model_to_safetensors('dummy_directory')

        # Assertions
        mock_save_file.assert_called_once()
        self.assertEqual(result, 'dummy_directory', "The function should return the directory path")

if __name__ == '__main__':
    unittest.main()
