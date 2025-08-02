

import pytest
import logging
from typing import Generator

from utils.database import Database
from utils.db_utils import DBUtils
from config.settings import Settings # Import Settings for type hinting

logger = logging.getLogger()

@pytest.fixture(scope="session")
def db_connection(config: Settings) -> Generator[Database, None, None]:
    """
    A session-scoped fixture that creates and manages the Database instance.
    It now depends on the `config` fixture to get settings at the right time.
    """
    db_settings = {
        'user': config.DB_USER,
        'password': config.DB_PASSWORD,
        'host': config.DB_HOST,
        'port': config.DB_PORT,
        'service_name': config.DB_SERVICE_NAME
    }

    if not all(db_settings.values()):
        pytest.fail("Database credentials are not fully configured.")

    db_instance = None
    try:
        db_instance = Database(db_settings)
        with db_instance.engine.connect() as connection:
            logger.info(f"Successfully established initial DB connection to: {db_settings['host']}")
        yield db_instance
    except Exception as e:
        pytest.fail(f"Database connection setup failed: {e}")
    finally:
        if db_instance:
            db_instance.close()

@pytest.fixture(scope="session")
def db_utils(db_connection: Database) -> DBUtils:
    """Provides a session-scoped instance of the DBUtils class."""
    return DBUtils(database=db_connection)
