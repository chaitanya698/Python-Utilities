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

class DBUtils:
def **init**(self, db_connection, logger):
self.db = db_connection
self.logger = logger

```
def verify_batch_chat_history_info_request(self, interaction_ids: List[str]) -> Dict[str, Any]:
    """
    Verify chat history info for multiple interaction IDs and generate a comprehensive log summary.
    
    Args:
        interaction_ids: List of interaction IDs to verify
        
    Returns:
        Dictionary containing verification results and summary statistics
    """
    self.logger.info(f"Starting batch verification for {len(interaction_ids)} interaction IDs")
    
    # Initialize result tracking
    verification_results = {
        'total_interactions': len(interaction_ids),
        'successful_verifications': 0,
        'failed_verifications': 0,
        'missing_entries': 0,
        'missing_impact_requests': 0,
        'parse_failures': 0,
        'validation_failures': 0,
        'detailed_results': {},
        'summary_by_error_type': {
            'no_entry_found': [],
            'missing_impact_request': [],
            'json_parse_error': [],
            'validation_error': [],
            'missing_required_fields': []
        }
    }
    
    # Required fields for validation
    required_fields = ['qName']  # Add other required fields as needed
    
    for interaction_id in interaction_ids:
        result = self._verify_single_interaction(interaction_id, required_fields)
        verification_results['detailed_results'][interaction_id] = result
        
        # Update counters based on result status
        if result['status'] == 'success':
            verification_results['successful_verifications'] += 1
        else:
            verification_results['failed_verifications'] += 1
            
            # Categorize the failure type
            error_type = result['error_type']
            verification_results['summary_by_error_type'][error_type].append(interaction_id)
            
            if error_type == 'no_entry_found':
                verification_results['missing_entries'] += 1
            elif error_type == 'missing_impact_request':
                verification_results['missing_impact_requests'] += 1
            elif error_type == 'json_parse_error':
                verification_results['parse_failures'] += 1
            elif error_type == 'validation_error':
                verification_results['validation_failures'] += 1
    
    # Generate and log comprehensive summary
    self._log_verification_summary(verification_results)
    
    return verification_results

def _verify_single_interaction(self, interaction_id: str, required_fields: List[str]) -> Dict[str, Any]:
    """
    Verify a single interaction's chat history info.
    
    Args:
        interaction_id: The interaction ID to verify
        required_fields: List of required field names
        
    Returns:
        Dictionary with verification result details
    """
    result = {
        'interaction_id': interaction_id,
        'status': None,
        'error_type': None,
        'missing_fields': [],
        'error_message': None,
        'data_summary': None
    }
    
    try:
        # Query the database for this interaction
        query_result = self.db.execute_query(
            "SELECT * FROM CHAT_HISTORY_INFO WHERE interaction_id = ?", 
            params={"interaction_id": interaction_id}
        )
        
        if not query_result:
            result.update({
                'status': 'failed',
                'error_type': 'no_entry_found',
                'error_message': f"No CHAT_HISTORY_INFO entry found for interaction ID: {interaction_id}"
            })
            self.logger.error(result['error_message'])
            return result
        
        # Check for impact_request blob
        request_blob = query_result[0].get('impact_request')
        if not request_blob:
            result.update({
                'status': 'failed',
                'error_type': 'missing_impact_request',
                'error_message': f"No 'impact_request' blob found in CHAT_HISTORY_INFO entry for {interaction_id}"
            })
            self.logger.error(result['error_message'])
            return result
        
        # Parse JSON data
        try:
            data = json.loads(request_blob)
            questions = data.get('questions', [])
            
            if not isinstance(questions, list):
                result.update({
                    'status': 'failed',
                    'error_type': 'validation_error',
                    'error_message': f"'questions' is not a list in request blob for {interaction_id}: {questions}"
                })
                self.logger.error(result['error_message'])
                return result
            
            # Extract qNames and find missing required fields
            qnames = {q.get('qName') for q in questions if isinstance(q, dict) and q.get('qName')}
            missing_fields = [field for field in required_fields if field not in qnames]
            
            if missing_fields:
                result.update({
                    'status': 'failed',
                    'error_type': 'missing_required_fields',
                    'missing_fields': missing_fields,
                    'error_message': f"Missing required qName(s) in CHAT_HISTORY_INFO for {interaction_id}: {missing_fields}",
                    'data_summary': {
                        'total_questions': len(questions),
                        'available_qnames': list(qnames),
                        'missing_qnames': missing_fields
                    }
                })
                self.logger.error(result['error_message'])
                return result
            
            # Success case
            result.update({
                'status': 'success',
                'error_type': None,
                'error_message': None,
                'data_summary': {
                    'total_questions': len(questions),
                    'available_qnames': list(qnames)
                }
            })
            self.logger.info(f"All required qNames are present in CHAT_HISTORY_INFO for {interaction_id}")
            return result
            
        except json.JSONDecodeError as e:
            result.update({
                'status': 'failed',
                'error_type': 'json_parse_error',
                'error_message': f"Failed to parse request blob as JSON for {interaction_id}: {e}"
            })
            self.logger.error(result['error_message'])
            return result
            
    except Exception as e:
        result.update({
            'status': 'failed',
            'error_type': 'validation_error',
            'error_message': f"Failed to verify CHAT_HISTORY_INFO for {interaction_id}: {e}"
        })
        self.logger.error(result['error_message'])
        return result

def _log_verification_summary(self, results: Dict[str, Any]) -> None:
    """
    Generate and log a comprehensive verification summary.
    
    Args:
        results: The verification results dictionary
    """
    total = results['total_interactions']
    successful = results['successful_verifications']
    failed = results['failed_verifications']
    
    # Main summary
    self.logger.info("=" * 80)
    self.logger.info("CHAT_HISTORY_INFO BATCH VERIFICATION SUMMARY")
    self.logger.info("=" * 80)
    self.logger.info(f"Total Interactions Processed: {total}")
    self.logger.info(f"Successful Verifications: {successful} ({successful/total*100:.1f}%)")
    self.logger.info(f"Failed Verifications: {failed} ({failed/total*100:.1f}%)")
    
    if failed > 0:
        self.logger.info("-" * 50)
        self.logger.info("FAILURE BREAKDOWN:")
        
        # Log each error type with counts and examples
        for error_type, interaction_list in results['summary_by_error_type'].items():
            if interaction_list:
                count = len(interaction_list)
                self.logger.info(f"  {error_type.replace('_', ' ').title()}: {count} interactions")
                
                # Show first few examples
                examples = interaction_list[:3]
                if len(interaction_list) > 3:
                    examples_str = f"{', '.join(examples)} (and {len(interaction_list)-3} more)"
                else:
                    examples_str = ', '.join(examples)
                self.logger.info(f"    Examples: {examples_str}")
        
        self.logger.info("-" * 50)
        self.logger.info("DETAILED MISSING FIELDS BY INTERACTION:")
        
        # Show detailed missing fields for interactions that have them
        for interaction_id, details in results['detailed_results'].items():
            if details['missing_fields']:
                missing_str = ', '.join(details['missing_fields'])
                self.logger.info(f"  {interaction_id}: Missing [{missing_str}]")
    
    self.logger.info("=" * 80)
    
    # Additional detailed logging for debugging
    if failed > 0:
        self.logger.debug("FULL DETAILED RESULTS:")
        for interaction_id, details in results['detailed_results'].items():
            if details['status'] == 'failed':
                self.logger.debug(f"ID: {interaction_id}")
                self.logger.debug(f"  Error Type: {details['error_type']}")
                self.logger.debug(f"  Message: {details['error_message']}")
                if details['missing_fields']:
                    self.logger.debug(f"  Missing Fields: {details['missing_fields']}")
                if details['data_summary']:
                    self.logger.debug(f"  Data Summary: {details['data_summary']}")
                self.logger.debug("-" * 30)
```

# Usage example:

def example_usage():
“””
Example of how to use the batch verification method
“””
# Assuming you have db_utils instance
db_utils = DBUtils(db_connection, logger)

```
# List of interaction IDs to verify
interaction_ids_to_check = [
    "interaction_001",
    "interaction_002", 
    "interaction_003",
    "interaction_004",
    "interaction_005"
]

# Run batch verification
verification_results = db_utils.verify_batch_chat_history_info_request(interaction_ids_to_check)

# You can also access specific results programmatically
failed_interactions = [
    interaction_id for interaction_id, details in verification_results['detailed_results'].items()
    if details['status'] == 'failed'
]

print(f"Failed interactions: {failed_interactions}")

# Get interactions with specific error types
missing_entries = verification_results['summary_by_error_type']['no_entry_found']
missing_fields_interactions = verification_results['summary_by_error_type']['missing_required_fields']

return verification_results
```

