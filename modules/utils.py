import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

def get_api_key(key_name="OPENROUTER_API_KEY"):
    """
    Get API key from environment variables

    """
    api_key = os.getenv(key_name)
    
    if not api_key:
        raise ValueError(f"Invalid API key: {key_name} not found in environment variables")
    
    return api_key

def initialize_llm(model_name="meta-llama/llama-3.3-70b-instruct",
                  temperature=0.4,
                  use_streaming=True):
    """
    Initialize LLM

    """
    api_key = get_api_key()
    callbacks = [StreamingStdOutCallbackHandler()]
    
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        streaming=use_streaming,
        callbacks=callbacks,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    return llm
