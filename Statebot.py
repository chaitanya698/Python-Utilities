import json
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Constants
STATE_FILE = "chat_state.json"

# Initialize LLM and memory
llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")

# === Utility for State Persistence ===

def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"state": "file_a_complaint"}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# === Chain Definitions ===

# 1. File a complaint
file_complaint_prompt = PromptTemplate(
    input_variables=["chat_history"],
    template="""
    {chat_history}
    ChatBOT: Would you like to file a complaint? (Yes/No)
    """
)
file_complaint_chain = LLMChain(llm=llm, prompt=file_complaint_prompt, memory=memory)

# 2. Initial Questions
initial_questions_prompt = PromptTemplate(
    input_variables=["chat_history"],
    template="""
    {chat_history}
    ChatBOT: Please answer the following:
    1. When was the complaint received?
    2. How was it received?
    3. What is the account number?
    """
)
initial_questions_chain = LLMChain(llm=llm, prompt=initial_questions_prompt, memory=memory)

# 3. Clarification Chain
clarification_prompt = PromptTemplate(
    input_variables=["chat_history"],
    template="""
    {chat_history}
    ChatBOT: Please clarify the issue in detail. Provide any supporting info.
    """
)
clarification_chain = LLMChain(llm=llm, prompt=clarification_prompt, memory=memory)

# 4. Classification Chain
classification_prompt = PromptTemplate(
    input_variables=["chat_history"],
    template="""
    {chat_history}
    ChatBOT: Based on the complaint, classify it into one of the categories:
    - Service Complaint
    - Product Complaint
    - Others
    """
)
classification_chain = LLMChain(llm=llm, prompt=classification_prompt, memory=memory)

# 5. Summary and Confirmation
summary_prompt = PromptTemplate(
    input_variables=["chat_history"],
    template="""
    {chat_history}
    ChatBOT: Here is the summary of your complaint. Please confirm or suggest revisions.
    """
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, memory=memory)

# === State Manager ===

def handle_state(state):
    if state == "file_a_complaint":
        response = file_complaint_chain.run({})
        if "yes" in response.lower():
            next_state = "initial_questions"
        else:
            next_state = "end"
    
    elif state == "initial_questions":
        response = initial_questions_chain.run({})
        next_state = "clarification"

    elif state == "clarification":
        response = clarification_chain.run({})
        next_state = "classification"

    elif state == "classification":
        response = classification_chain.run({})
        next_state = "summary"

    elif state == "summary":
        response = summary_chain.run({})
        if "confirm" in response.lower():
            next_state = "end"
        else:
            next_state = "clarification"

    else:
        print("ChatBOT: Ending session.")
        next_state = "end"

    print(response)
    return next_state

# === Main Loop with Resume Support ===

def run_chatbot():
    state_data = load_state()
    current_state = state_data.get("state", "file_a_complaint")

    while current_state != "end":
        try:
            next_state = handle_state(current_state)
            save_state({"state": next_state})
            current_state = next_state
        except Exception as e:
            print("ChatBOT: Encountered an unexpected error.")
            save_state({"state": current_state})
            break

    print("ChatBOT: Thank you. Goodbye!")

if __name__ == "__main__":
    run_chatbot()
