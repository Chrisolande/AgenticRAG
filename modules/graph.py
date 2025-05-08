from typing_extensions import TypedDict
from typing import List
from langgraph.graph import END, StateGraph
from pprint import pprint

from retriever import initialize_retriever
from chains import setup_chains, setup_web_search, process_web_search_results

# Define the structure of the graph state
class GraphState(TypedDict):
    question: str  # The input question
    generation: str  # The generated response
    web_search: str  # Indicator for web search usage
    documents: List[str]  # List of retrieved documents

# Function to retrieve documents from the vector store
def retrieve(state):
    """
    Retrieve documents from the vector store based on the input question.
    Args:
        state (dict): Current state containing the question.
    Returns:
        dict: Updated state with retrieved documents.
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Perform retrieval using the retriever
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

# Function to generate an answer using RAG
def generate(state):
    """
    Generate an answer using RAG based on retrieved documents.
    Args:
        state (dict): Current state containing the question and documents.
    Returns:
        dict: Updated state with the generated response.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # Generate response using the RAG chain
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# Function to grade the relevance of retrieved documents
def grade_documents(state):
    """
    Grade the relevance of retrieved documents to the input question.
    Args:
        state (dict): Current state containing the question and documents.
    Returns:
        dict: Updated state with filtered documents and web search flag.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Filter documents based on relevance
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

# Function to perform a web search based on the question
def web_search(state):
    """
    Perform a web search to retrieve additional information.
    Args:
        state (dict): Current state containing the question.
    Returns:
        dict: Updated state with web search results.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Perform web search and process results
    docs = web_search_tool.invoke({"query": question})
    web_results = process_web_search_results(docs)
    documents.append(web_results)
    return {"documents": documents, "question": question}

# Function to route the question to the appropriate workflow
def route_question(state):
    """
    Determine whether to use web search or vector store for the question.
    Args:
        state (dict): Current state containing the question.
    Returns:
        str: The next step in the workflow ("websearch" or "vectorstore").
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source['datasource'] == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

# Function to decide whether to generate an answer or perform a web search
def decide_to_generate(state):
    """
    Decide the next step based on the relevance of graded documents.
    Args:
        state (dict): Current state containing the web search flag and documents.
    Returns:
        str: The next step in the workflow ("websearch" or "generate").
    """
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    if web_search == "Yes":
        print("---DECISION: INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

# Function to grade the generated response for grounding and relevance
def grade_generation_v_documents_and_question(state):
    """
    Grade the generated response for grounding in documents and relevance to the question.
    Args:
        state (dict): Current state containing the question, documents, and generation.
    Returns:
        str: The next step in the workflow ("useful", "not useful", or "not supported").
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Check if the generation is grounded in the documents
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED---")
        return "not supported"

# Function to create and configure the LangGraph workflow
def create_graph():
    """
    Create and configure the LangGraph workflow.
    Returns:
        StateGraph: The compiled workflow graph.
    """
    workflow = StateGraph(GraphState)
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )
    return workflow.compile()

# Initialize global variables for graph components
def init_globals():
    """
    Initialize global variables needed for the graph.
    """
    global retriever, question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader, web_search_tool
    retriever = initialize_retriever(persist_directory="vector")
    question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader = setup_chains()
    web_search_tool = setup_web_search()

