import langchain, langchain_community
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=env_path)    
api_key = os.getenv("API_KEY")   

chat = ChatOpenAI(openai_api_key=api_key)

print(chat.invoke("Hi, How are you doing ?"))

