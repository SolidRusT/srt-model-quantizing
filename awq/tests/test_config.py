import unittest
from unittest.mock import patch
from app.config import Config, get_default_quanter

class TestConfig(unittest.TestCase):
    def test_setup_directories(self):
        with patch('os.makedirs') as mock_makedirs:
            Config.setup_directories()
            mock_makedirs.assert_any_call(Config.DATA_DIR, exist_ok=True)
            mock_makedirs.assert_any_call(Config.LOG_DIR, exist_ok=True)

    @patch('app.config.whoami')
    def test_get_default_quanter(self, mock_whoami):
        mock_whoami.return_value = {'name': 'test_user'}
        self.assertEqual(get_default_quanter(), 'test_user')

        mock_whoami.side_effect = Exception("API Error")
        self.assertIsNone(get_default_quanter())

if __name__ == '__main__':
    unittest.main()