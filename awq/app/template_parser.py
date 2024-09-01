import os
import logging

logger = logging.getLogger(__name__)

def read_template(template_path: str) -> str:
    """
    Read the content of a template file.
    """
    try:
        logger.info(f"Reading template file: {template_path}")
        with open(template_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logger.debug(f"Successfully read template file: {template_path}")
        return content
    except IOError as e:
        logger.error(f"Error reading template file {template_path}: {str(e)}")
        raise

def write_content_to_file(content: str, output_path: str) -> None:
    """
    Write content to a file.
    """
    try:
        logger.info(f"Writing content to file: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(content)
        logger.info(f"Content successfully written to {output_path}")
    except IOError as e:
        logger.error(f"Error writing to file {output_path}: {str(e)}")
        raise

def process_template(template_path: str, output_path: str, author: str, model: str) -> None:
    """
    Process a template file and write the result to an output file.
    """
    try:
        logger.info(f"Processing template: {template_path} for author: {author}, model: {model}")
        template_content = read_template(template_path)
        processed_content = template_content.replace("{AUTHOR}", author).replace("{MODEL}", model)
        write_content_to_file(processed_content, output_path)
        logger.info(f"Template processed successfully: {template_path} -> {output_path}")
    except Exception as e:
        logger.error(f"Error processing template: {str(e)}")
        raise
