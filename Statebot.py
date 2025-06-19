# chat_workflow.py

import logging
# Assuming DataLoaderService is imported from another module.
# from services import DataLoaderService 

logger = logging.getLogger(__name__)

class ChatWorkflow:
    """
    Manages the state and transitions of a complaint chat workflow.
    """
    def __init__(self, channel_app, conversation_id):
        """
        Initializes the workflow instance.
        """
        self.current_state = "complaint_initiation"
        self.channel_app = channel_app
        self.conversation_id = conversation_id
        self.prompt_template_id = None
        self.chat_prev_question = ""
        self.chat_prev_data = {}
        # As seen in your code, the number of revision attempts is set here.
        self.chat_clarification_revise_attempts = 5

    def transition(self, action, user_input):
        """
        Generic method to call the appropriate 'process' method based on the current state.
        """
        try:
            method_name = f"process_{self.current_state}"
            method = getattr(self, method_name)
            return method(action, user_input)
        except AttributeError:
            logger.error(f"No process method found for state: {self.current_state}")
            raise
        except Exception as e:
            logger.error(f"Error processing state {self.current_state}: {e}")
            raise e

    # =================================================================================
    # PROCESS METHODS (State to Transition Mapping)
    # Assembled from your images.
    # =================================================================================

    def process_complaint_initiation(self, action, user_input):
        return transition_complaint_initiation(self, action, user_input)

    def process_when_complaint_received(self, action, user_input):
        return transition_when_complaint_received(self, action, user_input)
        
    def process_how_complaint_received(self, action, user_input):
        return transition_how_complaint_received(self, action, user_input)

    def process_account_number_form(self, action, user_input):
        return transition_account_number_form(self, action, user_input)

    def process_account_number_question(self, action, user_input):
        return transition_account_number_question(self, action, user_input)

    def process_application_number_question(self, action, user_input):
        return transition_application_number_question(self, action, user_input)

    def process_reference_number_question(self, action, user_input):
        return transition_reference_number_question(self, action, user_input)

    def process_complaint_name(self, action, user_input):
        return transition_complaint_name(self, action, user_input)

    def process_complaint_description(self, action, user_input):
        return transition_complaint_description(self, action, user_input)

    def process_clarification_in_progress(self, action, user_input):
        return transition_clarification_in_progress(self, action, user_input)

    def process_clarification_summary(self, action, user_input):
        return transition_clarification_summary(self, action, user_input)

    def process_clarification_revise(self, action, user_input):
        return transition_clarification_revise(self, action, user_input)

    def process_clarification_revise_question(self, action, user_input):
        return transition_clarification_revise_question(self, action, user_input)
        
    def process_classification_summary(self, action, user_input):
        return transition_classification_summary(self, action, user_input)
    
    def process_unauthorized_account_question(self, action, user_input):
        return transition_unauthorized_account_question(self, action, user_input)
        
    def process_unauthorized_description(self, action, user_input):
        return transition_unauthorized_description(self, action, user_input)

    def process_preferred_communication_form(self, action, user_input):
        return transition_preferred_communication_form(self, action, user_input)

    def process_preferred_communication_question(self, action, user_input):
        return transition_preferred_communication_question(self, action, user_input)

    def process_create_complaint(self, action, user_input):
        return transition_create_complaint(self, action, user_input)
        
    def process_complaint_not_associated(self, action, user_input):
        return transition_complaint_not_associated(self, action, user_input)


    # =================================================================================
    # HELPER METHODS
    # Assembled from your images and our previous discussion.
    # =================================================================================
    
    def get_current_state_data(self):
        return DataLoaderService.get_wf_by_action_name(self.channel_app, self.current_state)
        
    def get_current_question(self):
        state_data = self.get_current_state_data()
        wf_question_id = state_data.get('chat_question_id', None)
        return DataLoaderService.get_chat_question_by_id(wf_question_id)

    def get_prev_question(self):
        return self.chat_prev_data.get('chat_question', None)
        
    def get_state_id(self):
        state_data = self.get_current_state_data()
        if state_data:
            return state_data["chat_workflow_action_id"]
        return None

    def get_state_data(self, state_id):
        return DataLoaderService.get_workflow_action_by_id(state_id)

    def get_current_state(self, state_id):
        state_data = self.get_state_data(state_id)
        if state_data:
            return state_data['action_name']
        return self.current_state

    def get_prompt_template_id(self, state_id):
        state_data = self.get_state_data(state_id)
        if state_data:
            return state_data.get('prompt_template_id', None)
        return None

    # +++ NEW METHOD +++
    # This is the new method to fetch the summary from the database.
    def get_latest_summary_from_history(self):
        """
        Fetches the latest summary text from the chat_history table for the current conversation.
        """
        logger.info(f"Fetching latest summary from history for conversation_id: {self.conversation_id}")
        try:
            # This requires a new method in your DataLoaderService (see previous response for the SQL).
            summary_record = DataLoaderService.get_latest_summary_record(self.conversation_id)

            if summary_record:
                summary_text = summary_record.get('CHAT_QUESTION')
                # Remove the "Final_Summary: " prefix from the text.
                if summary_text and summary_text.startswith("Final_Summary: "):
                    return summary_text.replace("Final_Summary: ", "", 1).strip()
                return summary_text
            
            logger.warning("No summary record was found in the chat history.")
            return None
        except Exception as e:
            logger.error(f"An error occurred while fetching summary from chat history: {e}")
            return None



# chat_workflow_service.py

import json
import logging
# Assuming these helper functions are defined elsewhere in your project.
# from utils import get_attempts

logger = logging.getLogger(__name__)


def populate_response_message(instance, message=None):
    """
    Creates the standard response object to be sent back to the user.
    """
    question_data = instance.get_current_question()
    question = message if message else question_data.get("question_text")
    logger.info(msg="Current question is: %s", question)
    metadata = question_data.get("question_metadata", {})
    response = json.loads(metadata)
    response["chatResponseText"] = question

    # The logic to remove the 'Modify' button was removed from this generic function
    # and moved into 'transition_clarification_summary' where there is more context.

    return response


def transition_clarification_summary(instance, action, user_input):
    """
    Handles the state transition after a summary has been presented to the user.
    The user can either 'continue' or 'modify'.
    """
    action_lower = action.lower() if action else ''

    if action_lower == 'continue':
        logger.info("User selected 'continue' after clarification summary.")
        instance.current_state = "classification_summary"
        return populate_response_message(instance)

    elif action_lower == 'modify':
        logger.info("User selected 'modify', composing clarify/revise question.")
        try:
            remaining_attempts = get_attempts(instance)
            if remaining_attempts > 0:
                instance.current_state = "clarification_revise_question"

                # Enhancement: Fetch summary from history with a fallback.
                summary_to_revise = ""
                try:
                    summary_to_revise = instance.get_latest_summary_from_history()
                    if not summary_to_revise:
                        logger.warning("Could not get summary from history. Falling back to chat_prev_data.")
                        summary_to_revise = instance.chat_prev_data.get("summary", "Could not retrieve the summary to revise.")
                except Exception as e:
                    logger.error(f"An error occurred while fetching the summary: {e}")
                    summary_to_revise = "Error: Could not retrieve the summary for revision."

                question_text = instance.get_current_question().get("question_text", "")
                question = question_text.replace("{}", str(remaining_attempts))
                
                response = populate_response_message(instance, message=question)
                
                # Add the fetched summary to the response payload for the frontend.
                response['data'] = response.get('data', {})
                response['data']['summary_to_revise'] = summary_to_revise

                # Enhancement: Remove 'Modify' button if this is the last attempt.
                if remaining_attempts == 1 and "actions" in response:
                    logger.info("Last modification attempt. Removing 'Modify' button for the next turn.")
                    response["actions"] = [
                        act for act in response["actions"]
                        if not (act.get("actionType") == "Button" and act.get("label") == "Modify")
                    ]
                
                attempt_counter = instance.chat_clarification_revise_attempts - remaining_attempts
                response['attemptCounter'] = attempt_counter
                return response
            else:
                # No attempts left, proceed to the next state.
                logger.warning("No remaining attempts for modification.")
                instance.current_state = "classification_summary"
                return populate_response_message(instance)
        except Exception as e:
            logger.exception(f"An unexpected error occurred in transition_clarification_summary 'modify' action: {e}")
            instance.current_state = "classification_summary"
            return populate_response_message(instance)
    else:
        logger.warning(f"Invalid action received: '{action}'. Defaulting to 'clarification_summary'.")
        instance.current_state = "clarification_summary"
        return populate_response_message(instance)


# services/data_loader.py

import logging
from sqlalchemy import create_engine, text, exc

logger = logging.getLogger(__name__)

# ==============================================================================
#  DATABASE ENGINE SETUP
# ==============================================================================
# This setup should be done once when your application starts.
#
# IMPORTANT: Replace the placeholder below with your actual database 
# connection string.
#
# --- Examples ---
# PostgreSQL: "postgresql+psycopg2://user:password@hostname:5432/database_name"
# MySQL:      "mysql+mysqlconnector://user:password@hostname:3306/database_name"
# MS SQL:     "mssql+pyodbc://user:password@your_dsn"
# ------------------------------------------------------------------------------
try:
    DATABASE_URL = "postgresql+psycopg2://user:password@localhost/space_redress_db"
    engine = create_engine(DATABASE_URL)
except ImportError:
    logger.critical("Database driver not installed. Please install a library like 'psycopg2-binary'.")
    engine = None
except Exception as e:
    logger.critical(f"Failed to create database engine: {e}")
    engine = None


class DataLoaderService:
    """
    A service class to handle all database interactions.
    All methods are static, so you don't need to instantiate the class.
    """

    @staticmethod
    def get_latest_summary_record(conversation_id: str):
        """
        Connects to the database and fetches the most recent summary record
        from the chat history for a given conversation.

        Args:
            conversation_id: The unique identifier for the conversation.

        Returns:
            A dictionary-like Row object representing the database record if found,
            otherwise returns None.
        """
        if not engine:
            logger.error("Database engine is not configured or available.")
            return None

        # The SQL query uses a named parameter ':conversation_id' for safety.
        # It finds records where the question indicates it's a summary,
        # orders them to get the latest one first, and takes only that one.
        query = text("""
            SELECT
                CHAT_QUESTION
            FROM
                space_redress.galaxy_chat_history
            WHERE
                chat_conversation_id = :conversation_id
                AND CHAT_QUESTION LIKE 'Final_Summary:%'
            ORDER BY
                CHAT_MSG_ORDER DESC
            LIMIT 1
        """)

        try:
            # The 'with' statement ensures the database connection is
            # automatically closed even if errors occur.
            with engine.connect() as connection:
                # Execute the query, safely passing the conversation_id.
                result = connection.execute(query, {"conversation_id": conversation_id})
                
                # .first() fetches the first row of the result or None if empty.
                record = result.first()
                
                if record:
                    logger.info(f"Successfully found latest summary for conversation '{conversation_id}'.")
                    # The 'record' is a SQLAlchemy Row object. It can be accessed by
                    # index or by column name, making it compatible with the
                    # .get('CHAT_QUESTION') call in your workflow class.
                    return record
                else:
                    logger.warning(f"No summary record found in history for conversation '{conversation_id}'.")
                    return None

        except exc.SQLAlchemyError as e:
            # Catches database-related errors (e.g., connection issues, syntax errors).
            logger.error(f"A database error occurred while fetching latest summary: {e}")
            return None
        except Exception as e:
            # Catches other unexpected errors.
            logger.error(f"An unexpected error occurred in get_latest_summary_record: {e}")
            return None

    # ==============================================================================
    #  PLACEHOLDERS FOR YOUR OTHER DATALOADER METHODS
    # ==============================================================================

    @staticmethod
    def get_wf_by_action_name(channel_app: str, current_state: str):
        # Your implementation for this function would go here.
        logger.debug(f"Called get_wf_by_action_name for state: {current_state}")
        # ... database logic ...
        return {"chat_question_id": 123} # Example return

    @staticmethod
    def get_chat_question_by_id(wf_question_id: int):
        # Your implementation for this function would go here.
        logger.debug(f"Called get_chat_question_by_id for ID: {wf_question_id}")
        # ... database logic ...
        return {"question_text": "This is a question from the DB.", "question_metadata": "{}"} # Example return

    @staticmethod
    def get_workflow_action_by_id(state_id: int):
        # Your implementation for this function would go here.
        logger.debug(f"Called get_workflow_action_by_id for ID: {state_id}")
        # ... database logic ...
        return {"action_name": "some_action", "prompt_template_id": 456} # Example return
