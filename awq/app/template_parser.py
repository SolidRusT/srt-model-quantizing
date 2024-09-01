import os
import logging

logger = logging.getLogger(__name__)

def read_template(template_path: str) -> str:
    """
    Read the content of a template file.
    """
    try:
        with open(template_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Template file not found: {template_path}")
        raise
    except IOError as e:
        logger.error(f"Error reading template file {template_path}: {str(e)}")
        raise

def process_template(template_path: str, output_path: str, author: str, model: str) -> None:
    """
    Process a template file and write the result to the output file.
    """
    try:
        content = read_template(template_path)
        
        # Replace placeholders
        content = content.replace('{AUTHOR}', author)
        content = content.replace('{MODEL}', model)
        
        # Write processed content to output file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        logger.info(f"Template processed and written to {output_path}")
    except Exception as e:
        logger.error(f"Error processing template {template_path}: {str(e)}")
        raise
