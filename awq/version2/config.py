import os

class Config:
    APP_HOME = os.getenv('APP_HOME', '/opt/quant-awq')
    QUANTER = 'solidrust'
    MODEL_REPO_DIR = os.path.join(APP_HOME, 'repos', 'srt-model-quantizing', 'awq')
    DATA_DIR = os.path.join(APP_HOME, 'data')
    LOG_FILE = os.path.join(APP_HOME, 'logs', 'quant-awq.log')
