import os
from dotenv import load_dotenv

def load_config():
    """Loads configuration from the .env file for the specified environment."""
    env = os.getenv('ENV', 'dev') # Default to 'dev' if not set
    env_path = os.path.join('config', f'{env}.env')
    load_dotenv(dotenv_path=env_path)
    return {
        'base_url': os.getenv('BASE_URL'),
        'client_id': os.getenv('CLIENT_ID'),
        'timeout': int(os.getenv('API_TIMEOUT', 30))
    }