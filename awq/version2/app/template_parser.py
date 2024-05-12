import os
from app.config import Config
import logging

def read_template(template_path):
    """
    Reads a template file from the specified path.

    Args:
        template_path (str): Path to the template file.

    Returns:
        str: Content of the template file.
    """
    with open(template_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_content_to_file(content, file_path):
    """
    Writes content to a file at the specified path.

    Args:
        content (str): Content to write to the file.
        file_path (str): Path to the file where content should be written.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    logging.info(f"Content written to {file_path}")

def process_template(author, model, template_path, output_path):
    """
    Processes a template file by replacing placeholders with specific values and writes to the output path.

    Args:
        author (str): Author name to replace in the template.
        model (str): Model name to replace in the template.
        template_path (str): Path to the template file.
        output_split_path (str): Path to save the processed template.
    """
    try:
        content = read_template(template_path)
        content = content.format(AUTHOR=author, MODEL=model)
        write_content_to_file(content, output_path)
        logging.info(f"Template processed and saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to process template {template_path}: {e}")
        raise
