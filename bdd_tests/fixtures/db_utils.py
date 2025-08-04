import logging
from sqlalchemy import text
from typing import List, Dict, Any

# Import the Database class for type hinting
from utils.database_util import Database

logger = logging.getLogger(__name__)


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
    
    def verify_conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists in the database."""
        query = text("""
            SELECT COUNT(*) as count 
            FROM galaxy_complaint_ai.chat_history 
            WHERE conversation_id = :conversation_id
        """)
        
        try:
            with self.db.get_session() as session:
                result = session.execute(query, {"conversation_id": conversation_id})
                count = result.scalar()
                exists = count > 0 if count else False
                logger.info(f"Conversation {conversation_id} exists: {exists}")
                return exists
        except Exception as e:
            logger.error(f"Failed to verify conversation: {e}")
            return False
    
    def get_complaint_details(self, interaction_id: str) -> Dict[str, Any]:
        """Retrieve complaint details by interaction ID."""
        query = text("""
            SELECT 
                interaction_id,
                conversation_id,
                complaint_date,
                complaint_method,
                account_number,
                complaint_details,
                summary,
                contact_willingness,
                status,
                created_at,
                updated_at
            FROM galaxy_complaint_ai.complaints 
            WHERE interaction_id = :interaction_id
        """)
        
        try:
            with self.db.get_session() as session:
                result = session.execute(query, {"interaction_id": interaction_id})
                row = result.first()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to retrieve complaint details: {e}")
            return None