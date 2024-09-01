import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the parent directory to the Python path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model_utils import authenticate_huggingface, download_model, check_model_files, convert_pytorch_to_safetensors, get_model_size, validate_model_checksum, find_file
from app.config import Config

class TestModelUtils(unittest.TestCase):

    @patch('app.model_utils.login')
    def test_authenticate_huggingface_success(self, mock_login):
        with patch.dict(os.environ, {'HF_ACCESS_TOKEN': 'test_token'}):
            authenticate_huggingface()
            mock_login.assert_called_once_with('test_token')

    @patch('app.model_utils.login')
    def test_authenticate_huggingface_no_token(self, mock_login):
        with patch.dict(os.environ, {}, clear=True):
            authenticate_huggingface()
            mock_login.assert_not_called()

    @patch('app.model_utils.snapshot_download')
    def test_download_model_success(self, mock_snapshot_download):
        mock_snapshot_download.return_value = '/path/to/model'
        result = download_model('author', 'model')
        self.assertEqual(result, '/path/to/model')
        mock_snapshot_download.assert_called_once()

    @patch('app.model_utils.snapshot_download')
    def test_download_model_failure(self, mock_snapshot_download):
        mock_snapshot_download.side_effect = Exception('Download failed')
        with self.assertRaises(Exception):
            download_model('author', 'model')

    @patch('app.model_utils.find_file')
    def test_check_model_files_valid(self, mock_find_file):
        mock_find_file.side_effect = ['/path/to/config.json', '/path/to/tokenizer.json', '/path/to/pytorch_model.bin']
        self.assertTrue(check_model_files('/path/to/model'))

    @patch('app.model_utils.find_file')
    def test_check_model_files_invalid(self, mock_find_file):
        mock_find_file.side_effect = ['/path/to/config.json', '', '']
        self.assertFalse(check_model_files('/path/to/model'))

    @patch('os.walk')
    def test_get_model_size(self, mock_walk):
        mock_walk.return_value = [
            ('/path/to/model', [], ['file1.bin', 'file2.bin']),
        ]
        with patch('os.path.getsize', side_effect=[1000000, 2000000]):
            size = get_model_size('/path/to/model')
            self.assertEqual(size, 3000000)

    def test_validate_model_checksum(self):
        # This is a placeholder test and should be updated with actual checksum logic
        self.assertTrue(validate_model_checksum('/path/to/model', 'expected_checksum'))

    @patch('os.walk')
    @patch('os.path.islink')
    @patch('os.path.realpath')
    def test_find_file(self, mock_realpath, mock_islink, mock_walk):
        mock_walk.return_value = [
            ('/path/to/model', [], ['config.json', 'model.bin']),
        ]
        mock_islink.return_value = False
        result = find_file('/path/to/model', 'config.json')
        self.assertEqual(result, '/path/to/model/config.json')

if __name__ == '__main__':
    unittest.main()
