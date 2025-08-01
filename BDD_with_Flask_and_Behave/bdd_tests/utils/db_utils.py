# bdd_tests/utils/db_utils.py

import logging
from sqlalchemy import text
from typing import List, Dict, Any

from bdd_tests.config.db_config import Session

logger = logging.getLogger(__name__)

class DBUtils:
    """
    A lightweight utility class that runs queries using the shared SQLAlchemy Session.
    """
    def get_chat_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Queries the database for all entries for a given conversation ID using a managed session.
        """
        sql_query = text("""
            SELECT * FROM galaxy_complaint_ai.chat_history 
            WHERE conversation_id = :conv_id 
            ORDER BY timestamp ASC;
        """)
        
        logger.info(f"Querying chat history for conversation_id: {conversation_id}")
        
        with Session() as session:
            try:
                result = session.execute(sql_query, {"conv_id": conversation_id})
                results_list = [dict(row) for row in result.mappings()]
                logger.info(f"Found {len(results_list)} records for conversation_id: {conversation_id}")
                return results_list
            except Exception as e:
                logger.error(f"Database query failed for conversation_id {conversation_id}: {e}")
                session.rollback()
                raise
