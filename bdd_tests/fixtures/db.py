# fixtures/db.py

import pytest
import logging
from typing import Generator

# Import the new Database manager and the central config object
from utils.database import Database
from utils.db_utils import DBUtils
from config.config_loader import config

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def db_connection() -> Generator[Database, None, None]:
    """
    A session-scoped fixture that creates and manages the Database instance.
    
    This fixture is responsible for:
    1. Creating a single `Database` instance for the entire test session using
       credentials from the central config.
    2. Yielding the instance to be used by other fixtures.
    3. Ensuring the database connection pool is closed after all tests run.
    """
    db_settings = {
        'user': config.db.user,
        'password': config.db.password,
        'host': config.db.host,
        'port': config.db.port,
        'service_name': config.db.service_name
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
    
    This fixture depends on the `db_connection` fixture and initializes
    the utility class with the live database connection manager.
    """
    return DBUtils(database=db_connection)
