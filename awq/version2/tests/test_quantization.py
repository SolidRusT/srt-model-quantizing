import unittest
from unittest.mock import patch, MagicMock
from app.quantization import run_quantization

class TestQuantization(unittest.TestCase):
    def setUp(self):
        # Set up a mock logger
        self.mock_logger = MagicMock()

    @patch('app.quantization.subprocess.run')
    def test_run_quantization_success(self, mock_run):
        # Configure the mock to not raise any exception
        mock_run.return_value = None

        # Call the function
        run_quantization("dummy_model_path", "dummy_config", self.mock_logger)

        # Check if subprocess.run was called correctly
        mock_run.assert_called_once_with(['quantize_model_tool', '--model', 'dummy_model_path', '--config', 'dummy_config'], check=True)

        # Assert logging was called with info
        self.mock_logger.info.assert_called_once_with("Quantization successful")

    @patch('app.quantization.subprocess.run')
    def test_run_quantization_failure(self, mock_run):
        # Configure the mock to raise a CalledProcessError
        mock_run.side_effect = subprocess.CalledProcessError(1, 'quantize_model_tool')

        # Test to ensure the exception is raised
        with self.assertRaises(subprocess.CalledProcessError):
            run_quantization("dummy_model_path", "dummy_config", self.mock_logger)

        # Assert logging was called with error
        self.mock_logger.error.assert_called_once_with("Quantization failed")

if __name__ == '__main__':
    unittest.main()
