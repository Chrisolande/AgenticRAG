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

def setup_chains():
    """
    Set up chains for question routing, RAG generation, and grading

    """
    # Initialize LLM
    llm = initialize_llm()
    
    # Question router chain
    question_router = ROUTER_PROMPT | llm | JsonOutputParser()
    
    # RAG chain
    rag_chain = GENERATION_PROMPT | llm | StrOutputParser()
    
    # Retrieval grader chain
    retrieval_grader = RETRIEVAL_GRADER_PROMPT | llm | JsonOutputParser()
    
    # Hallucination grader chain
    hallucination_grader = HALLUCINATION_GRADER_PROMPT | llm | JsonOutputParser()
    
    # Answer grader chain
    answer_grader = ANSWER_GRADER_PROMPT | llm | JsonOutputParser()
    
    return question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader

def setup_web_search():
    """
    Set up web search tool

    """
    # Get Tavily API key from environment variables
    os.environ['TAVILY_API_KEY'] = get_api_key("TAVILY_API_KEY")
    
    # Initialize web search tool
    web_search_tool = TavilySearchResults(k=3)
    
    return web_search_tool

def process_web_search_results(docs):
    """
    Process web search results into a Document

    """
    web_results = "\n".join([d["content"] for d in docs])
    return Document(page_content=web_results)
