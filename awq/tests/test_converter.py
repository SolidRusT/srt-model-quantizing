import unittest
from unittest.mock import patch, mock_open
import torch
from app.converter import convert_model_to_safetensors

class TestConverter(unittest.TestCase):
    @patch('app.converter.glob.glob')
    @patch('app.converter.torch.load')
    @patch('app.converter.save_file')
    def test_convert_model_to_safetensors(self, mock_save_file, mock_torch_load, mock_glob):
        # Setup
        mock_glob.return_value = ['model.bin']
        mock_torch_load.return_value = {'weights': torch.tensor([1.0, 2.0, 3.0])}

        # Test action
        result = convert_model_to_safetensors('dummy_directory')

        # Assertions
        mock_save_file.assert_called_once()
        self.assertEqual(result, 'dummy_directory', "The function should return the directory path")

    @patch('app.converter.glob.glob')
    def test_convert_model_to_safetensors_no_files(self, mock_glob):
        # Setup
        mock_glob.return_value = []

        # Test action
        result = convert_model_to_safetensors('dummy_directory')

        # Assertions
        self.assertEqual(result, 'dummy_directory', "The function should return the directory path even if no files are found")

if __name__ == '__main__':
    unittest.main()
