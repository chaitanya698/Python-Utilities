# bdd_tests/utils/db_utils.py

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Any

from bdd_tests.config.settings import settings

logger = logging.getLogger(__name__)

class DBUtils:
    """
    A utility class to handle all database interactions using SQLAlchemy.
    This provides a robust, session-managed approach to DB validation.
    """
    def __init__(self):
        """
        Initializes the SQLAlchemy engine and creates a session factory.
        The engine is created once and is designed to be thread-safe.
        """
        try:
            # Create an engine using the connection URL from settings
            self.engine = create_engine(settings.DATABASE_URL, echo=False)
            # Create a configured "Session" class
            self.Session = sessionmaker(bind=self.engine)
            logger.info("SQLAlchemy engine and session maker initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SQLAlchemy engine: {e}")
            raise

    def get_chat_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Queries the database for all entries for a given conversation ID using a managed session.

        Args:
            conversation_id: The unique ID of the conversation to retrieve.

        Returns:
            A list of dictionaries, where each dictionary represents a row from the chat history.
        """
        # Define the SQL query using sqlalchemy.text() to parameterize inputs safely
        sql_query = text("""
            SELECT * FROM galaxy_complaint_ai.chat_history 
            WHERE conversation_id = :conv_id 
            ORDER BY timestamp ASC;
        """)
        
        logger.info(f"Querying chat history for conversation_id: {conversation_id}")
        
        # Use a 'with' statement to ensure the session is properly closed
        with self.Session() as session:
            try:
                result = session.execute(sql_query, {"conv_id": conversation_id})
                # .mappings() provides a dictionary-like interface for each row
                results_list = [dict(row) for row in result.mappings()]
                logger.info(f"Found {len(results_list)} records for conversation_id: {conversation_id}")
                return results_list
            except Exception as e:
                logger.error(f"Database query failed for conversation_id {conversation_id}: {e}")
                session.rollback() # Rollback the transaction on error
                raise
