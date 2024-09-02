import unittest
from unittest.mock import patch, MagicMock
from app.model_utils import authenticate_huggingface, download_model, check_model_files

class TestModelUtils(unittest.TestCase):
    @patch('app.model_utils.login')
    @patch('app.model_utils.os.environ.get')
    def test_authenticate_huggingface(self, mock_env_get, mock_login):
        mock_env_get.return_value = 'test_token'
        result = authenticate_huggingface()
        mock_login.assert_called_once_with('test_token')
        self.assertEqual(result, 'test_token')

    @patch('app.model_utils.snapshot_download')
    def test_download_model(self, mock_snapshot_download):
        mock_snapshot_download.return_value = '/path/to/model'
        result = download_model('author', 'model')
        self.assertEqual(result, '/path/to/model')

    @patch('app.model_utils.find_file')
    def test_check_model_files(self, mock_find_file):
        mock_find_file.side_effect = [True, True, True]  # Simulate finding all required files
        self.assertTrue(check_model_files('/path/to/model'))

        mock_find_file.side_effect = [True, False, True]  # Simulate missing a required file
        self.assertFalse(check_model_files('/path/to/model'))

if __name__ == '__main__':
    unittest.main()