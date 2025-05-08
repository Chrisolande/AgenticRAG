from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
import os

from utils import initialize_llm, get_api_key
from prompts import (
    ROUTER_PROMPT, 
    GENERATION_PROMPT, 
    RETRIEVAL_GRADER_PROMPT,
    HALLUCINATION_GRADER_PROMPT,
    ANSWER_GRADER_PROMPT
)

# Function to set up various chains for RAG workflows
def setup_chains():
    """
    Set up chains for question routing, RAG generation, and grading.
    Returns initialized chains for different tasks.
    """
    # Initialize the language model (LLM)
    llm = initialize_llm()
    
    # Define a chain for routing questions to appropriate handlers
    question_router = ROUTER_PROMPT | llm | JsonOutputParser()
    
    # Define a chain for retrieval-augmented generation (RAG)
    rag_chain = GENERATION_PROMPT | llm | StrOutputParser()
    
    # Define a chain for grading the quality of retrieved documents
    retrieval_grader = RETRIEVAL_GRADER_PROMPT | llm | JsonOutputParser()
    
    # Define a chain for evaluating hallucinations in generated responses
    hallucination_grader = HALLUCINATION_GRADER_PROMPT | llm | JsonOutputParser()
    
    # Define a chain for grading the final answers
    answer_grader = ANSWER_GRADER_PROMPT | llm | JsonOutputParser()
    
    return question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader

# Function to set up a web search tool
def setup_web_search():
    """
    Set up web search tool using Tavily API.
    Returns an initialized web search tool.
    """
    # Get Tavily API key from environment variables
    os.environ['TAVILY_API_KEY'] = get_api_key("TAVILY_API_KEY")
    
    # Initialize the web search tool with a limit of 3 results
    web_search_tool = TavilySearchResults(k=3)
    
    return web_search_tool

# Function to process web search results into a Document object
def process_web_search_results(docs):
    """
    Process web search results into a Document object.
    Args:
        docs (list): List of dictionaries containing web search results.
    Returns:
        Document: A single Document object containing concatenated search results.
    """
    # Combine the content of all search results into a single string
    web_results = "\n".join([d["content"] for d in docs])
    return Document(page_content=web_results)
