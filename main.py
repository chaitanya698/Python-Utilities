# main.py
import logging
import requests
import json
import os
from flask import Flask, request, jsonify, g # Import g for request context
from app.repository.database_import import load_prompt_template_data # Assuming this path is correct

# Import the exception handler registration function
from exception_handler import register_exception_handlers 

# Import the ChatWorkflow class
from chat_workflow import ChatWorkflow

# Configure logging (good practice to configure globally or via config)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# --- Configuration Loading ---
# Function for Profiles/Configs
def load_config(env):
    """
    Loads application configuration based on the environment.
    """
    if env == "development":
        # Assuming config_local.Config and config_cloud.Config exist
        # and contain the necessary configuration objects.
        import config.config_local as config_module
        app.config.from_object(config_module.Config)
    else:
        import config.config_cloud as config_module
        app.config.from_object(config_module.Config)
    logger.info(f"Configuration loaded for environment: {env}")

# --- Global Data Initialization ---
PROMPT_TEMPLATE_DICT = {}

# Function to initialize PROMPT_TEMPLATE_DICT
def initialize_prompt_template_dict():
    """
    Load prompt templates within the application context.
    """
    with app.app_context():
        # Temporarily not loading as the tables doesn't exist in QA UAT env.
        # This part assumes load_prompt_template_data() is blocking or handles its own errors.
        # prompt_template_dict = load_prompt_template_data()
        # For now, just initialize as empty if not loading from DB
        global PROMPT_TEMPLATE_DICT
        PROMPT_TEMPLATE_DICT = {} # Set to empty dict as per your comment
        
        if not PROMPT_TEMPLATE_DICT: # Using `if not PROMPT_TEMPLATE_DICT` is better than `len(...) == 0`
            logger.error("PROMPT_TEMPLATE_DICT is empty. Please check the database connection and data.")
        else:
            logger.info(f"Loaded {len(PROMPT_TEMPLATE_DICT)} prompt templates from the database.")
    return PROMPT_TEMPLATE_DICT

# --- Register Exception Handlers ---
# This needs to be done *after* the app is created.
register_exception_handlers(app)
logger.info("Global exception handlers registered.")

# --- Initialize Chat Workflow ---
chat_workflow = ChatWorkflow()
logger.info("ChatWorkflow instance created.")


# --- Application Setup Function ---
def create_app():
    """
    Central function to create and configure the Flask application.
    """
    env = os.getenv('FLASK_ENV', 'development')
    load_config(env)

    # Set SQLALCHEMY_DATABASE_URI using DB_URL from the config object
    # app.config['SQLALCHEMY_DATABASE_URI'] = app.config['DB_URL']

    # Initialize the database with the app (if you have a db.init_app)
    # from app.repository.database_init import db
    # db.init_app(app)

    # Call the function to initialize PROMPT_TEMPLATE_DICT.
    # PROMPT_TEMPLATE_DICT can be used globally across the application.
    initialize_prompt_template_dict()
    
    return app

# --- Routes ---

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    API endpoint for the chatbot interaction.
    Expects a JSON payload with 'user_input'.
    """
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({"status": "error", "message": "Missing 'user_input' in request."}), 400

    logger.info(f"Received user input: '{user_input}'")

    # Process the transition and update the workflow state
    chat_workflow.process_transition(user_input)

    # Get the structured response data
    response_data = chat_workflow.get_response_data()

    # Add request_id to the response payload for traceability
    response_data["request_id"] = getattr(g, 'request_id', 'N/A')

    # Determine HTTP status code based on workflow status
    http_status_code = 200
    if response_data.get("status") == "error":
        # If the chat workflow itself entered an error state, use 500 or the error_enum's http_status
        # For simplicity, if chat_workflow.py marks it as an error, it's generally a 500 for the API client
        # unless a specific error code mapping dictates otherwise.
        http_status_code = 500 
        # Note: If ComplaintException is caught by global handler, it will set status code.
        # This is for errors caught *within* the chat_workflow.process_transition.

    logger.info(f"Sending response: {response_data}")
    return jsonify(response_data), http_status_code

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}), 200


# --- Application Startup ---
if __name__ == '__main__':
    # Call create_app to configure and get the app instance
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)