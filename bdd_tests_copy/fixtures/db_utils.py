import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..database.db_manager import DatabaseManager
from ..utils.logger_config import get_logger


class DBUtils:
    """Database utility functions for test automation."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize with database manager."""
        self.db = db_manager
        self.logger = get_logger(__name__)
    
    def get_chat_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve chat history for a conversation."""
        query = """
            SELECT 
                conversation_id,
                message_id,
                timestamp,
                user_message,
                bot_response,
                action,
                status
            FROM galaxy_complaint_ai.chat_history 
            WHERE conversation_id = :conversation_id 
            ORDER BY timestamp ASC
        """
        
        self.logger.info(f"Retrieving chat history for conversation: {conversation_id}")
        
        try:
            results = self.db.execute_query(query, {"conversation_id": conversation_id})
            self.logger.info(f"Found {len(results)} messages for conversation: {conversation_id}")
            return results
        except Exception as e:
            self.logger.error(f"Failed to retrieve chat history: {e}")
            return []
    
    def verify_conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists in the database."""
        query = """
            SELECT COUNT(*) as count 
            FROM galaxy_complaint_ai.chat_history 
            WHERE conversation_id = :conversation_id
        """
        
        try:
            results = self.db.execute_query(query, {"conversation_id": conversation_id})
            exists = results[0]['count'] > 0 if results else False
            self.logger.info(f"Conversation {conversation_id} exists: {exists}")
            return exists
        except Exception as e:
            self.logger.error(f"Failed to verify conversation: {e}")
            return False
    
    def get_complaint_details(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complaint details by interaction ID."""
        query = """
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
        """
        
        try:
            results = self.db.execute_query(query, {"interaction_id": interaction_id})
            return results[0] if results else None
        except Exception as e:
            self.logger.error(f"Failed to retrieve complaint details: {e}")
            return None
    
    def get_test_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get test execution metrics for reporting."""
        query = """
            SELECT 
                COUNT(DISTINCT conversation_id) as total_conversations,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                AVG(CASE 
                    WHEN completion_time IS NOT NULL 
                    THEN (completion_time - start_time) * 24 * 60 * 60
                END) as avg_duration_seconds
            FROM galaxy_complaint_ai.test_executions
            WHERE start_time BETWEEN :start_date AND :end_date
        """
        
        try:
            results = self.db.execute_query(query, {
                "start_date": start_date,
                "end_date": end_date
            })
            return results[0] if results else {
                "total_conversations": 0,
                "completed": 0,
                "failed": 0,
                "avg_duration_seconds": 0
            }
        except Exception as e:
            self.logger.error(f"Failed to retrieve test metrics: {e}")
            return {}
    
    def cleanup_test_data(self, conversation_ids: List[str]) -> bool:
        """Clean up test data after test execution."""
        if not conversation_ids:
            return True
        
        try:
            # Clean up chat history
            delete_chat_query = """
                DELETE FROM galaxy_complaint_ai.chat_history 
                WHERE conversation_id IN :conversation_ids
            """
            
            with self.db.get_session() as session:
                session.execute(delete_chat_query, {"conversation_ids": tuple(conversation_ids)})
                session.commit()
            
            self.logger.info(f"Cleaned up test data for {len(conversation_ids)} conversations")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clean up test data: {e}")
            return False