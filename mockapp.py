import streamlit as st
import os
import dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

dotenv.load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

MODELS = [
    "google/gemini-2.5-pro-exp-03-25",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemini-2.0-flash-exp:free"
]

st.set_page_config(
    page_title="RAG LLM app?", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)