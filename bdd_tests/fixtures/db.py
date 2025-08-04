import pytest
import logging
from typing import Generator

# Import the Database manager and config
from utils.database_util import Database
from fixtures.db_utils import DBUtils
from config.loader import get_config

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def db_connection() -> Generator[Database, None, None]:
    """
    A session-scoped fixture that creates and manages the Database instance.
    """
    config = get_config()
    
    db_settings = {
        'user': config.DB_USER,
        'password': config.DB_PASSWORD,
        'host': config.DB_HOST,
        'port': config.DB_PORT,
        'service_name': config.DB_SERVICE_NAME
    }
    
    # Fail fast if credentials are not fully configured
    if not all(db_settings.values()):
        pytest.fail(
            "Database credentials are not fully configured. "
            "Please provide them via command line or environment variables."
        )
    
    db_instance = None
    try:
        db_instance = Database(db_settings)
        # Test the connection to ensure it's valid before running tests
        with db_instance.engine.connect() as connection:
            logger.info(f"Successfully established initial DB connection to: {db_settings['host']}")
        
        # Yield the instance to the tests
        yield db_instance
        
    except Exception as e:
        logger.critical(f"Failed to create and verify database connection: {e}")
        pytest.fail(f"Database connection setup failed: {e}")
    
    finally:
        if db_instance:
            logger.info("Tearing down database connection at end of session.")
            db_instance.close()


@pytest.fixture(scope="session")
def db_utils(db_connection: Database) -> DBUtils:
    """
    Provides a session-scoped instance of the DBUtils class.
    """
    return DBUtils(database=db_connection)