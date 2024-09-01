import unittest
from unittest.mock import patch, MagicMock
from app.model_utils import setup_environment, download_model, check_pytorch_files
from app.config import Config

class TestModelUtils(unittest.TestCase):
    @patch('app.model_utils.os.makedirs')
    def test_setup_environment(self, mock_makedirs):
        # Test that the environment setup creates necessary directories
        setup_environment('author', 'model')
        mock_makedirs.assert_called_with(Config.DATA_DIR, exist_ok=True)

    @patch('app.model_utils.os.path.exists')
    @patch('app.model_utils.requests.get')
    def test_download_model(self, mock_get, mock_exists):
        # Setup
        mock_exists.return_value = False
        mock_get.return_value = MagicMock(status_code=200, content=b'model data')

        # Action
        result = download_model('author', 'model')

        # Assert
        self.assertEqual(result, f"{Config.DATA_DIR}/model")
        mock_get.assert_called_once()

    @patch('app.model_utils.glob.glob')
    def test_check_pytorch_files(self, mock_glob):
        # Setup
        mock_glob.return_value = ['path/to/pytorch_model.bin']

        # Test check_pytorch_files when files are present
        model_path = 'dummy/path'
        result = check_pytorch_files(model_path)
        self.assertTrue(result)
        mock_glob.assert_called_with(f'{model_path}/**/*.bin', recursive=True)

        # Test check_pytorch_files when no files are found
        mock_glob.return_value = []
        result = check_pytorch_to_safetensors_files(model_path)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
