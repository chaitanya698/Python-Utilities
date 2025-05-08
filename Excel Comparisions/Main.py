import requests
import json
import os
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, abort
from app import app
from app.repository.db_file import db, ma
from app.chat.search_service import SearchService
from app.common.util import Util
from app.chat.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChatService_logger")

# Define the database connection string
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
service_name = os.getenv("DB_SERVICE_NAME")

# Global variables
variable = 'CASEDESCRIPTION'
session_manager = SessionManager()

# Function for Profiles/Configs
def load_config(env):
    if env == "development":
        app.config.from_object('config.config_dev.Config')
        # Construct the SQLALCHEMY_DATABASE_URI
        app.config['SQLALCHEMY_DATABASE_URI'] = f"oracle+cx_oracle://{user}:{password}@{host}:{port}/{service_name}"
    else:
        app.config.from_object('config.config_prod.Config')

# Set environment
env = os.getenv("FLASK_ENV", "development")
load_config(env)

# Callhome endpoint
def callhome():
    data = {
        "applicationMetadata": {
            "applicationName": "complaints-chat",
            "distributedId": "space",
            "componentName": "space",
            "groupId": "space",
            "artifactId": "space-complaints-chat",
            "description": f"{request.language.lower_case} application",
            "artifactVersion": "1.0",
            "applicationType": "python",
            "frameworkVersion": "python",
            "springProfile": "dev",
            "orchestraBomVersion": "2.3",
            "authXFilterEnabled": "false",
            "dependencies": [{"groupname":"javax.xml.bind"},{"dependencyName":"jaxb-api"}],
            "resilienceData": "",
            "lastExecutedDateTime": 1670522168794,
            "createdDateTime": 1670522168794
        },
        "satisfactionCriteria": [
            {
                "criteria": "Other",
                "detail": []
            }
        ],
        "applicationVersion": "1.0"
    }
    
    url = "https://ca02726-orchestra-callhome-service-dev.apps.cic-lmr-n-01.cf.wellsfargo.net/api/application"
    json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, data=json_data, headers=headers, verify=False)
        response.raise_for_status()
        return response.status_code
    except requests.exceptions.RequestException as e:
        print(e, 'request failed')

class ChatService:
    def __init__(self, payload, session_manager):
        self.payload = payload
        self.chat_text = payload['chatText']
        self.session_manager = session_manager
        self.session_id = payload['sessionId']
    
    def initialize_chat(self):
        try:
            # Generate session id and create history
            self.session_id = Util.generate_session_id()
            self.session_manager.create_session(self.session_id)
            print(f"Session ID: {self.session_id}")
            
            self.session_manager.add_message_to_session(self.session_id, self.chat_text, 
                'Welcome to Complaint Capture Assistant')
            
            return jsonify({
                'sessionId': self.session_id,
                'chatResponse': 'Welcome to Complaint Capture Assistant, please provide the details of your complaint.',
                'enableYesOrNo': 'No'
            })
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            abort(500, description="An unexpected error occurred processing complaint determination. Please try again later.")
    
    def generate_complaint_type(self):
        try:
            if not self.session_manager.validate_session(self.session_id):
                abort(400, description="Invalid session ID.")
            
            print("Start file read complaint type rules")
            complaintType_rules_input_file_path = './config/prompts/ComplaintTypeRules.txt'
            rules_content = Util.replace_variable_in_file(complaintType_rules_input_file_path,
                                                      variable,
                                                      self.chat_text + " " + self.session_manager.get_last_message(self.session_id))
            print("End file read complaint type rules")
            
            # Use ThreadPoolExecutor for async processing
            with ThreadPoolExecutor() as executor:
                future_rules = executor.submit(Util.process_chat_completion, self.session_id, rules_content)
                rulesContentResponse = future_rules.result()
                
                self.session_manager.add_message_to_session(self.session_id, self.payload['chatText'], rulesContentResponse)
                
                return jsonify({
                    'chatResponse': rulesContentResponse,
                    'sessionId': self.payload['sessionId'],
                    'enableYesOrNo': 'No'
                })
        except FileNotFoundError:
            print(f'ComplaintTypeRules File not found: {complaintType_rules_input_file_path}')
            abort(500, description="An unexpected error occurred processing complaint determination. Please try again later.")
        except Exception as e:
            logger.error(f"An unexpected error occurred processing complaint determination: {e}")
            abort(500, description="An unexpected error occurred processing complaint determination. Please try again later.")

    def generate_summary(self):
        try:
            if not self.session_manager.validate_session(self.session_id):
                abort(400, description="Invalid session ID.")
            
            print("Start file read Summarization")
            complaint_summarization_input_file_path = './config/prompts/ComplaintSummarization.txt'
            summary_content = Util.replace_variable_in_file(complaint_summarization_input_file_path,
                                                        variable,
                                                        self.chat_text + " " + self.session_manager.get_last_message(self.session_id))
            print("End file read Summarization")
            
            # Use ThreadPoolExecutor for async processing
            with ThreadPoolExecutor() as executor:
                future_summary = executor.submit(Util.process_chat_completion, self.session_id, summary_content)
                summaryResponse = future_summary.result()
                
                self.session_manager.add_message_to_session(self.session_id, self.payload['chatText'], summaryResponse)
                
                return jsonify({
                    'chatResponse': summaryResponse,
                    'sessionId': self.payload['sessionId'],
                    'enableYesOrNo': 'No'
                })
        except FileNotFoundError:
            print(f'ComplaintSummarization File not found: {complaint_summarization_input_file_path}')
            abort(500, description="An unexpected error occurred processing complaint summary. Please try again later.")
        except Exception as e:
            logger.error(f"An unexpected error occurred processing complaint summary: {e}")
            abort(500, description="An unexpected error occurred processing complaint summary. Please try again later.")

    def extract_fields(self):
        try:
            if not self.session_manager.validate_session(self.session_id):
                abort(400, description="Invalid session ID.")
            
            print("Start file read fields extraction")
            complaint_fields_extraction_input_file_path = './config/prompts/ComplaintFieldsExtraction.txt'
            fields_extraction_content = Util.replace_variable_in_file(complaint_fields_extraction_input_file_path,
                                                                 variable,
                                                                 self.chat_text + " " + self.session_manager.get_last_message(self.session_id))
            print("End file read fields extraction")
            
            extracted_fields = Util.process_chat_completion(self.session_id, fields_extraction_content)
            #print(session_data)  # Shows both sessions and their message history
            
            self.session_manager.add_message_to_session(self.session_id, self.payload['chatText'], extracted_fields)
            
            return jsonify({
                'chatResponse': extracted_fields,
                'sessionId': self.payload['sessionId'],
                'enableYesOrNo': 'Yes'
            })
        except Exception as e:
            logger.error(f"An unexpected error occurred processing complaint field extraction: {e}")
            abort(500, description="Field extraction error")

    def submit_complaint(self):
        try:
            if not self.session_manager.validate_session(self.session_id):
                abort(400, description="Invalid session ID.")
            
            print("Start submit complaint")
            #Submit the complaint
            complaint_id = Util.generate_human_readable_complaint_id()
            
            return jsonify({
                'chatResponse': f"Great! Complaint#{complaint_id} submitted with WellsFargo!",
                'sessionId': self.payload['sessionId'],
                'enableYesOrNo': 'No'
            })
        except Exception as e:
            logger.error(f"An unexpected error occurred processing complaint submission: {e}")
            abort(500, description="error")

# Define routes
@app.route('/chat', methods=['POST'])
def chat():
    payload = request.json
    action = payload.get('action', 'initialize_chat')
    
    chat_service = ChatService(payload, session_manager)
    
    # Route to the appropriate method based on action
    if action == 'initialize_chat':
        return chat_service.initialize_chat()
    elif action == 'generate_complaint_type':
        return chat_service.generate_complaint_type()
    elif action == 'generate_summary':
        return chat_service.generate_summary()
    elif action == 'extract_fields':
        return chat_service.extract_fields()
    elif action == 'submit_complaint':
        return chat_service.submit_complaint()
    else:
        abort(400, description=f"Unknown action: {action}")

if __name__ == '__main__':
    db.init_app(app)
    ma.init_app(app)
    app.run(debug=False)
    # callhome()
