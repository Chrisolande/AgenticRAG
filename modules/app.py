import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, SystemMessage
from typing import List

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

class PARAMETERS:
    model_name = "deepseek/deepseek-prover-v2:free"
    streaming = True
    temperature = 0.1
    top_p = 0.95
    max_tokens = 2000
    seed = 42

# Custom streaming callback handler for Gradio
class StreamingGradioCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        
    def get_response(self):
        return self.text

system_prompt = """
You are an intelligent and friendly AI assistant. Be clear, concise, and helpful in your responses. 
Adapt your tone to match the user's style—formal when they are formal, casual when they are casual. 
If the user asks a question that's ambiguous, politely ask for clarification. 
If something requires code, examples, or step-by-step explanations, provide them as needed. 
Avoid speculation, and say "I don't know" when you're not sure. Always be brief and give short summaries unless it is a mathematical question.
Do not leave your answers with latex on them, always try to implement them in a readable format for the human

Always prioritize user understanding, and be patient and encouraging in your tone.
"""

def format_message(message:str, history:List[tuple]):
    if len(history) > 5:
        history = history[-5:] # Keep the last 5, take care of context length
    
    messages = [SystemMessage(content=system_prompt)]
    
    # Add history
    for user_msg, model_answer in history:
        messages.append(HumanMessage(content=user_msg))
        messages.append(SystemMessage(content=model_answer))
    
    # Add current message
    messages.append(HumanMessage(content=message))
    
    return messages

def predict(message, history):
    messages = format_message(message, history)
    
    # Create a new callback handler for each request
    streaming_handler = StreamingGradioCallbackHandler()
    
    llm = ChatOpenAI(
        model_name=PARAMETERS.model_name,
        temperature=PARAMETERS.temperature,
        streaming=PARAMETERS.streaming,
        callbacks=[streaming_handler],
        top_p=PARAMETERS.top_p,
        max_tokens=PARAMETERS.max_tokens,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    # Start the generation process
    llm.invoke(messages)
    
    # For streaming in Gradio, we need to yield partial responses
    partial_response = ""
    for token in streaming_handler.text:
        partial_response += token
        yield partial_response

gr.ChatInterface(predict).queue().launch()