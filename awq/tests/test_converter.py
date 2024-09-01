import unittest
from unittest.mock import patch, mock_open, MagicMock
import torch
from app.converter import convert_model_to_safetensors, save_file  # Add save_file import

class TestConverter(unittest.TestCase):
    @patch('app.converter.glob.glob')
    @patch('app.converter.torch.load')
    @patch('app.converter.safetensors_save_file')
    @patch('app.converter.os.remove')
    @patch('app.converter.os.path.exists')
    def test_convert_model_to_safetensors(self, mock_exists, mock_remove, mock_save_file, mock_load, mock_glob):
        # Setup
        mock_glob.return_value = ['model.bin']
        mock_load.return_value = {'weights': torch.tensor([1.0, 2.0, 3.0])}
        mock_exists.return_value = False

        # Test action
        result = convert_model_to_safetensors('dummy_directory')

        # Assertions
        mock_load.assert_called_once_with('model.bin', map_location='cpu')
        mock_save_file.assert_called_once_with({'weights': torch.tensor([1.0, 2.0, 3.0])}, 'model.safetensors')
        mock_remove.assert_called_once_with('model.bin')
        self.assertEqual(result, 'dummy_directory')

    @patch('app.converter.glob.glob')
    @patch('app.converter.torch.load')
    @patch('app.converter.safetensors_save_file')
    @patch('app.converter.os.remove')
    @patch('app.converter.os.path.exists')
    def test_convert_model_to_safetensors_multiple_files(self, mock_exists, mock_remove, mock_save_file, mock_load, mock_glob):
        # Setup
        mock_glob.return_value = ['model_part1.bin', 'model_part2.bin']
        mock_load.return_value = {'weights': torch.tensor([1.0, 2.0, 3.0])}
        mock_exists.return_value = False

        # Test action
        result = convert_model_to_safetensors('dummy_directory')

        # Assertions
        self.assertEqual(mock_load.call_count, 2)
        self.assertEqual(mock_save_file.call_count, 2)
        self.assertEqual(mock_remove.call_count, 2)
        self.assertEqual(result, 'dummy_directory')

    @patch('app.converter.safetensors_save_file')
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
