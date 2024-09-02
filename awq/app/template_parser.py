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

def process_template(template_path: str, output_path: str, **kwargs):
    """
    Process a template file and write the result to an output file.

    Args:
        template_path (str): Path to the template file.
        output_path (str): Path where the processed file will be written.
        **kwargs: Keyword arguments to be replaced in the template.
    """
    try:
        logger.info(f"Processing template: {template_path}")
        with open(template_path, 'r', encoding='utf-8') as file:
            template_content = file.read()

        # Replace placeholders in the template
        for key, value in kwargs.items():
            placeholder = f"{{{key.upper()}}}"
            template_content = template_content.replace(placeholder, str(value))

        # Write the processed content to the output file
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(template_content)

        logger.info(f"Template processed successfully: {template_path} -> {output_path}")
    except Exception as e:
        logger.error(f"Error processing template: {str(e)}")
        raise
