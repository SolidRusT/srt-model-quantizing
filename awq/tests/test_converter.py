import unittest
from unittest.mock import patch, mock_open, MagicMock
import torch
from app.converter import convert_model_to_safetensors, save_file

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

    @patch('app.converter.glob.glob')
    @patch('app.converter.torch.load')
    @patch('app.converter.save_file')
    def test_convert_model_to_safetensors_multiple_files(self, mock_save_file, mock_torch_load, mock_glob):
        # Setup
        mock_glob.return_value = ['model_part1.bin', 'model_part2.bin']
        mock_torch_load.side_effect = [
            {'weights1': torch.tensor([1.0, 2.0])},
            {'weights2': torch.tensor([3.0, 4.0])}
        ]

        # Test action
        result = convert_model_to_safetensors('dummy_directory')

        # Assertions
        self.assertEqual(mock_save_file.call_count, 2, "save_file should be called for each model part")
        self.assertEqual(result, 'dummy_directory', "The function should return the directory path")

    @patch('safetensors.torch.save_file')
    def test_save_file(self, mock_safetensors_save):
        # Setup
        tensors = {'weight': torch.tensor([1.0, 2.0, 3.0])}
        filename = 'test_model.safetensors'

        # Test action
        save_file(tensors, filename)

        # Assertions
        mock_safetensors_save.assert_called_once_with(tensors, filename)

    @patch('app.converter.glob.glob')
    @patch('app.converter.torch.load')
    def test_convert_model_to_safetensors_error_handling(self, mock_torch_load, mock_glob):
        # Setup
        mock_glob.return_value = ['model.bin']
        mock_torch_load.side_effect = RuntimeError("Failed to load model")

        # Test action and assertion
        with self.assertRaises(RuntimeError):
            convert_model_to_safetensors('dummy_directory')

if __name__ == '__main__':
    unittest.main()
