import os
from config import Config

def read_template(file_path):
    """Read the template file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(file_path, content):
    """Write content to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def update_template(template, author, model):
    """Replace placeholders in the template with actual values."""
    return template.format(AUTHOR=author, MODEL=model)

def process_template(source_path, dest_path, author, model):
    """Read, update, and write template content to a new location."""
    template = read_template(source_path)
    updated_content = update_template(template, author, model)
    write_file(dest_path, updated_content)
    print(f"Updated content written to {dest_path}")
