import oracledb
import logging
from bdd_tests.config.settings import settings

# Configure logger for the database manager
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages the connection to the Oracle database.
    This class handles creating and closing the database connection pool.
    """
    _pool = None

    @classmethod
    def initialize_pool(cls):
       
        if cls._pool:
            logger.info("Database pool is already initialized.")
            return

        try:
            # The "not a registered listener" error is almost always due to an
            # incorrect DSN. The `oracledb.makedsn` function helps build it correctly.
            dsn = oracledb.makedsn(
                settings.DB_HOST,
                settings.DB_PORT,
                service_name=settings.DB_SERVICE_NAME
            )
            logger.info(f"Attempting to connect with DSN: {dsn}")

            # Create a connection pool. This is more efficient than creating
            # new connections for every test.
            cls._pool = oracledb.create_pool(
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                dsn=dsn,
                min=2,
                max=5,
                increment=1,
                encoding="UTF-8"
            )
            logger.info("Database connection pool initialized successfully.")
        except oracledb.Error as e:
            logger.error(f"Failed to initialize Oracle database pool: {e}")
            # Re-raise the exception to fail the test run immediately if DB is not available.
            raise

    @classmethod
    def get_connection(cls):

        if not cls._pool:
            logger.error("Database pool is not initialized. Call initialize_pool() first.")
            raise Exception("Database pool not initialized.")
        
        try:
            connection = cls._pool.acquire()
            logger.info("Acquired a database connection from the pool.")
            return connection
        except oracledb.Error as e:
            logger.error(f"Failed to acquire connection from pool: {e}")
            raise

    @classmethod
    def close_pool(cls):
        """
        Closes the connection pool and releases all resources.
        This should be called once when the test suite finishes.
        """
        if cls._pool:
            try:
                cls._pool.close()
                cls._pool = None
                logger.info("Database connection pool closed successfully.")
            except oracledb.Error as e:
                logger.error(f"Error closing the database connection pool: {e}")
                raise
