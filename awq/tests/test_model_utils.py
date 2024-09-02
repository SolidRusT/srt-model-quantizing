import unittest
from unittest.mock import patch, MagicMock
from app.model_utils import authenticate_huggingface, download_model, check_model_files, find_file, calculate_checksum, convert_pytorch_to_safetensors, get_model_size, validate_model_checksum, calculate_directory_checksum

class TestModelUtils(unittest.TestCase):
    @patch('app.model_utils.login')
    @patch('app.model_utils.os.environ.get')
    @patch('app.model_utils.HfFolder.get_token')
    def test_authenticate_huggingface(self, mock_get_token, mock_env_get, mock_login):
        # Test when token is in environment variables
        mock_env_get.return_value = 'test_token'
        mock_get_token.return_value = None
        result = authenticate_huggingface()
        mock_login.assert_called_once_with('test_token')
        self.assertEqual(result, 'test_token')

        # Test when token is not in environment variables but in HfFolder
        mock_env_get.return_value = None
        mock_get_token.return_value = 'hf_token'
        result = authenticate_huggingface()
        mock_login.assert_called_with('hf_token')
        self.assertEqual(result, 'hf_token')

        # Test when token is not found anywhere
        mock_env_get.return_value = None
        mock_get_token.return_value = None
        result = authenticate_huggingface()
        self.assertIsNone(result)

    @patch('app.model_utils.snapshot_download')
    def test_download_model(self, mock_snapshot_download):
        mock_snapshot_download.return_value = '/path/to/model'
        result = download_model('author', 'model')
        self.assertEqual(result, '/path/to/model')

        # Test exception handling
        mock_snapshot_download.side_effect = Exception("Download failed")
        with self.assertRaises(Exception):
            download_model('author', 'model')

    @patch('app.model_utils.find_file')
    def test_check_model_files(self, mock_find_file):
        # Test when all required files are found (single file model)
        mock_find_file.side_effect = [True, True, True, False, False]
        self.assertTrue(check_model_files('/path/to/model'))

        # Test when all required files are found (sharded safetensors model)
        mock_find_file.side_effect = [True, True, False, True, False]
        self.assertTrue(check_model_files('/path/to/model'))

        # Test when all required files are found (sharded PyTorch model)
        mock_find_file.side_effect = [True, True, False, False, True]
        self.assertTrue(check_model_files('/path/to/model'))

        # Test when a required file is missing
        mock_find_file.side_effect = [True, False]
        self.assertFalse(check_model_files('/path/to/model'))

        # Test when no valid model weights are found
        mock_find_file.side_effect = [True, True, False, False, False]
        self.assertFalse(check_model_files('/path/to/model'))

    @patch('os.path.exists')
    @patch('os.walk')
    def test_find_file(self, mock_walk, mock_exists):
        mock_walk.return_value = [('/path/to/model', [], ['file.txt'])]
        mock_exists.return_value = True
        result = find_file('/path/to/model', 'file.txt')
        self.assertEqual(result, '/path/to/model/file.txt')

        mock_walk.return_value = [('/path/to/model', [], ['other.txt'])]
        result = find_file('/path/to/model', 'file.txt')
        self.assertEqual(result, '')

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=b'test data')
    @patch('hashlib.md5')
    def test_calculate_checksum(self, mock_md5, mock_open):
        mock_md5.return_value.hexdigest.return_value = 'test_checksum'
        result = calculate_checksum('/path/to/file')
        self.assertEqual(result, 'test_checksum')

    @patch('os.listdir')
    @patch('os.path.join')
    @patch('torch.load')
    @patch('app.model_utils.safetensors_save_file')
    @patch('os.remove')
    def test_convert_pytorch_to_safetensors(self, mock_remove, mock_safetensors_save_file, mock_torch_load, mock_join, mock_listdir):
        mock_listdir.return_value = ['model.bin', 'model.pt', 'config.json']
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_torch_load.return_value = {'key': 'value'}
        
        convert_pytorch_to_safetensors('/path/to/model')
        
        mock_safetensors_save_file.assert_any_call({'key': 'value'}, '/path/to/model/model.safetensors')
        mock_safetensors_save_file.assert_any_call({'key': 'value'}, '/path/to/model/model.safetensors')
        self.assertEqual(mock_remove.call_count, 2)

    @patch('os.walk')
    @patch('os.path.getsize')
    def test_get_model_size(self, mock_getsize, mock_walk):
        mock_walk.return_value = [
            ('/path/to/model', [], ['file1.txt', 'file2.txt']),
            ('/path/to/model/subdir', [], ['file3.txt'])
        ]
        mock_getsize.side_effect = [1024 * 1024, 2 * 1024 * 1024, 3 * 1024 * 1024]
        result = get_model_size('/path/to/model')
        self.assertEqual(result, 6 * 1024 * 1024)

    @patch('app.model_utils.calculate_directory_checksum')
    def test_validate_model_checksum(self, mock_calculate_checksum):
        mock_calculate_checksum.return_value = 'test_checksum'
        self.assertTrue(validate_model_checksum('/path/to/model', 'test_checksum'))
        self.assertFalse(validate_model_checksum('/path/to/model', 'wrong_checksum'))

    @patch('os.walk')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=b'test data')
    @patch('hashlib.sha256')
    def test_calculate_directory_checksum(self, mock_sha256, mock_open, mock_walk):
        mock_walk.return_value = [
            ('/path/to/model', [], ['file1.txt', 'file2.txt']),
            ('/path/to/model/subdir', [], ['file3.txt'])
        ]
        mock_sha256.return_value.hexdigest.return_value = 'test_checksum'
        result = calculate_directory_checksum('/path/to/model')
        self.assertEqual(result, 'test_checksum')

if __name__ == '__main__':
    unittest.main()