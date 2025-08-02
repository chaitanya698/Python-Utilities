# utils/database.py

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.url import URL
from typing import Dict

logger = logging.getLogger(__name__)

class Database:
    """
    A class to manage the lifecycle of the database connection engine and sessions.
    This encapsulates the database configuration and setup logic.
    """
    def __init__(self, db_settings: Dict[str, any]):
        """
        Initializes the Database manager.

        Args:
            db_settings (Dict[str, any]): A dictionary containing database credentials
                                          (user, password, host, port, service_name).
        """
        self.engine = None
        self.SessionFactory = None
        self._db_settings = db_settings
        self._initialize()

    def _initialize(self):
        """
        Constructs the database URL and creates the SQLAlchemy engine and session factory.
        """
        if not all(self._db_settings.values()):
            raise ValueError("All database settings (user, password, host, port, service_name) are required.")

        try:
            database_url = str(URL.create(
                drivername="oracle+oracledb",
                username=self._db_settings['user'],
                password=self._db_settings['password'],
                host=self._db_settings['host'],
                port=self._db_settings['port'],
                database=self._db_settings['service_name'],
            ))

            engine_options = {
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30,
                "pool_recycle": 1800,
                "pool_pre_ping": True,
            }

            self.engine = create_engine(database_url, **engine_options)
            
            # Create a configured "Session" class
            self.SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            logger.info(f"Database engine created for host: {self._db_settings['host']}")

        except Exception as e:
            logger.critical(f"Failed to initialize SQLAlchemy engine: {e}")
            raise

    def get_session(self):
        """
        Provides a new SQLAlchemy session from the session factory.
        This should be used in a 'with' statement to ensure it's closed properly.
        
        Returns:
            A new SQLAlchemy Session object.
        """
        if not self.SessionFactory:
            raise RuntimeError("SessionFactory is not initialized. Call _initialize() first.")
        return self.SessionFactory()

    def close(self):
        """
        Disposes of the engine's connection pool.
        Should be called at the end of the test session.
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine connection pool disposed.")

