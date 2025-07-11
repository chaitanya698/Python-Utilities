
import os
from dotenv import load_dotenv

def get_config():
    """
    Loads configuration from the relevant .env file.
    The environment can be switched by setting the 'ENV' os variable.
    e.g., ENV=qa behave
    Defaults to 'dev' if not set.
    """
    env = os.getenv('ENV', 'dev')
    dotenv_path = f'.env.{env}'
    load_dotenv(dotenv_path=dotenv_path)
    
    config = {
        'API_BASE_URL': os.getenv('API_BASE_URL'),
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO')
    }
    return config