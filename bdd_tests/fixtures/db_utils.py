

import logging
from sqlalchemy import text
from typing import List, Dict, Any

# Import the new Database class for type hinting
from .database_util import Database

logger = logging.getLogger()

class DBUtils:
    """
    A utility class that uses a managed Database instance to run queries.
    """
    def __init__(self, database: Database):
        """
        Initializes the DBUtils with a Database connection manager.

        Args:
            database (Database): An initialized instance of the Database class.
        """
        self.db = database

    def get_chat_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Queries the database for chat history using a session from the connection manager.
        """
        sql_query = text("""
            SELECT * FROM galaxy_complaint_ai.chat_history 
            WHERE conversation_id = :conv_id 
            ORDER BY timestamp ASC
        """)
        
        logger.info(f"Querying chat history for conversation_id: {conversation_id}")
        
        # Get a new session from our database manager for this specific operation
        with self.db.get_session() as session:
            try:
                result = session.execute(sql_query, {"conv_id": conversation_id})
                results_list = [dict(row) for row in result.mappings()]
                logger.info(f"Found {len(results_list)} records for conversation_id: {conversation_id}")
                return results_list
            except Exception as e:
                logger.error(f"Database query failed for conversation_id {conversation_id}: {e}")
                session.rollback()
                raise
