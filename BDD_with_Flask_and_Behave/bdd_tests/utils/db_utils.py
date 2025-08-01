# bdd_tests/utils/db_utils.py

import logging
# Use a placeholder for the database driver to avoid installation errors if not needed.
# The user should replace this with the actual driver (e.g., psycopg2, pyodbc).
try:
    import psycopg2
    from psycopg2.extras import DictCursor
except ImportError:
    psycopg2 = None 

from typing import List, Dict, Any
from bdd_tests.config.settings import settings

logger = logging.getLogger(__name__)

class DBUtils:
    """A utility class to handle all database interactions for test validation."""

    def __init__(self):
        """Initializes the DB connection details from the global settings."""
        if not psycopg2:
            raise ImportError("psycopg2 is not installed. Please install it to use DBUtils.")
            
        self.conn_params = {
            'host': settings.DB_HOST,
            'port': settings.DB_PORT,
            'user': settings.DB_USER,
            'password': settings.DB_PASSWORD,
            'dbname': settings.DB_NAME
        }
        self.connection = None

    def connect(self):
        """Establishes a connection to the database."""
        try:
            logger.info(f"Connecting to database '{self.conn_params['dbname']}' on host '{self.conn_params['host']}'.")
            self.connection = psycopg2.connect(**self.conn_params)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def disconnect(self):
        """Closes the database connection if it's open."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed.")

    def get_chat_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Queries the database for all entries in the chat_history table for a given conversation ID.
        
        Args:
            conversation_id: The unique ID of the conversation to retrieve.

        Returns:
            A list of dictionaries, where each dictionary represents a row from the chat history.
        """
        if not self.connection:
            self.connect()

        query = "SELECT * FROM galaxy_complaint_ai.chat_history WHERE conversation_id = %s ORDER BY timestamp ASC;"
        logger.info(f"Executing query to fetch chat history for conversation_id: {conversation_id}")
        
        results = []
        try:
            with self.connection.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(query, (conversation_id,))
                results = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Found {len(results)} records for conversation_id: {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to execute query for conversation_id {conversation_id}: {e}")
            raise
        
        return results
