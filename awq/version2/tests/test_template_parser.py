import unittest
from unittest.mock import patch, mock_open
from app.template_parser import read_template, write_content_to_file, process_template

class TestTemplateParser(unittest.TestCase):

    def test_read_template(self):
        # Setup a mock open that reads specific content
        m = mock_open(read_data="content with {AUTHOR} and {MODEL}")
        with patch('builtins.open', m):
            content = read_templatetemplatee('/fake/path/template.md')

        # Assert content is read correctly
        self.assertEqual(content, "content with {AUTHOR} and {MODEL}")

    def test_write_content_to_file(self):
        # Setup a mock open
        m = mock_open()
        with patch('builtins.open', m):
            write_content_to_file("some content", '/fake/path/output.md')

        # Check if the write function was called correctly
        m.assert_called_once_with('/fake/path/output.md', 'w', encoding='utf-8')
        handle = m()
        handle.write.assert_called_once_with("some content")

    @patch('app.template_parser.read_template')
    @patch('app.template_parser.write_content_to_file')
    def test_process_template(self, mock_write, mock_read):
        # Setup the mocks
        mock_read.return_value = "Model created by {AUTHOR}, using {MODEL}."
        author = "Jane"
        model = "AI-Model"

        # Run the function under test
        process_template(author, model, '/fake/path/template.md', '/fake/path/output.md')

        # Check if the read and write functions were called correctly
        mock_read.assert_called_once_with('/fake/path/template.md')
        mock_write.assert_called_once_with("Model created by Jane, using AI-Model.", '/fake/path/output.md')

if __name__ == '__main__':
    unittest.main()
