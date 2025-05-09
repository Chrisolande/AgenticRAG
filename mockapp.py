import streamlit as st
import uuid
import os
import traceback  # For detailed error reporting
from typing import List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from modules.utils import get_api_key, initialize_llm
from modules.retriever import initialize_retriever, create_vectorstore, load_documents
from modules.graph import create_graph, init_globals
from modules.chains import setup_chains, setup_web_search

# Set environment variables to prevent PyTorch errors
os.environ["PYTORCH_JIT"] = "0"

# Streamlit page configuration
st.set_page_config(
    page_title="Agentic RAG App",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 <i>Agentic RAG</i> 🤖💬</h2>""")

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state['rag_sources'] = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! How may I help you today? Try uploading a .txt file and asking about its content (e.g., 'Frankenstein')."}
    ]

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Sidebar ---
with st.sidebar:
    # API Key Handling
    try:
        openrouter_api_key = get_api_key("OPENROUTER_API_KEY")
        st.session_state.openrouter_api_key = openrouter_api_key
        # Test LLM initialization
        try:
            test_llm = initialize_llm()
            st.success("OpenRouter API key validated!")
        except Exception as e:
            st.error(f"Failed to validate OpenRouter API key: {e}")
    except ValueError:
        default_api_key = os.getenv("OPENROUTER_API_KEY", "")
        with st.popover("🔐 OpenRouter API Key"):
            openrouter_api_key = st.text_input(
                "Enter your OpenRouter API Key",
                value=default_api_key,
                type="password",
                key="openrouter_api_key"
            )
            if openrouter_api_key:
                os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
                st.session_state.openrouter_api_key = openrouter_api_key

    st.divider()
    # Model Selection
    model = st.selectbox(
        "Select a model",
        options=["meta-llama/llama-3.3-70b-instruct"],
        key="model"
    )

    # Toggles and Buttons
    cols = st.columns(3)
    with cols[0]:
        st.toggle(
            "Use RAG",
            value=st.session_state.get("use_rag", False),
            key="use_rag",
            disabled=not st.session_state.get("openrouter_api_key")
        )
    with cols[1]:
        st.toggle(
            "Verbose",
            value=False,
            key="verbose"
        )
    with cols[2]:
        st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

    # File Upload for RAG
    st.header("RAG Resources")
    uploaded_files = st.file_uploader(
        "Upload text documents",
        type="txt",
        accept_multiple_files=True,
        key="rag_docs"
    )

    # Process Uploaded Files
    if uploaded_files:
        new_documents = []
        for file in uploaded_files:
            try:
                file_contents = file.getvalue().decode("utf-8")
                doc = Document(page_content=file_contents, metadata={"filename": file.name})
                if file_contents not in [d.page_content for d in st.session_state.rag_sources]:
                    new_documents.append(doc)
            except Exception as e:
                st.error(f"Failed to process file {file.name}: {e}")
        if new_documents:
            st.session_state.rag_sources.extend(new_documents)
            # Create or update vector store
            try:
                st.session_state.vector_db = create_vectorstore(
                    documents=st.session_state.rag_sources,
                    store_type="faiss",
                    persist_directory="vector"
                )
                st.success(f"Vector store updated with {len(new_documents)} new documents!")
                # Verify index files
                if os.path.exists("vector/index.faiss"):
                    st.write("FAISS index saved successfully.")
                else:
                    st.warning("FAISS index not found in 'vector' directory.")
            except Exception as e:
                st.error(f"Failed to update vector store: {e}")

    # Display Documents in DB
    with st.expander(f"Documents in DB ({len(st.session_state.rag_sources)})"):
        for doc in st.session_state.rag_sources:
            st.write(f"- {doc.metadata.get('filename', 'Unnamed Document')}")

    # Debug: Check Vector Store
    if st.session_state.vector_db:
        st.write(f"Vector store loaded: {type(st.session_state.vector_db).__name__}")
        try:
            # Test retriever
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 2})
            test_docs = retriever.invoke("test query")
            st.write(f"Retriever test: {len(test_docs)} documents retrieved.")
        except Exception as e:
            st.error(f"Retriever test failed: {e}")
    else:
        st.warning("No vector store loaded. Upload documents or check 'vector' directory.")

# --- Main Chat App ---
if st.session_state.get("openrouter_api_key") and 'model' in st.session_state:
    # Initialize LLM
    try:
        llm = initialize_llm(model_name=st.session_state.model)
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        llm = None

    # Initialize RAG if enabled and documents are available
    if st.session_state.use_rag and (st.session_state.rag_sources or os.path.exists("vector")) and llm:
        if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
            try:
                init_globals()  # Initialize retriever, chains, and web search
                st.session_state.rag_chain = create_graph()
                st.success("RAG pipeline initialized!")
            except Exception as e:
                st.error(f"Failed to initialize RAG pipeline: {e}")

    def handle_message(message: str) -> str:
        """Handles a user message and returns the assistant's response."""
        if message.lower() in ["exit", "quit"]:
            st.session_state.messages.append({"role": "user", "content": message})
            st.session_state.messages.append({"role": "assistant", "content": "Exiting chat. Goodbye!"})
            return "Exiting chat. Goodbye!"

        st.session_state.messages.append({"role": "user", "content": message})
        
        response = "No response generated."
        
        if st.session_state.use_rag and st.session_state.rag_chain:
            try:
                # Use a spinner to show progress
                with st.spinner('Processing with RAG...'):
                    try:
                        # IMPORTANT: Use invoke() instead of stream() to avoid PyTorch errors
                        result = st.session_state.rag_chain.invoke({"question": message})
                        
                        # Extract the final generation
                        response = result.get("generation", "No response generated.")
                        
                        # Verbose mode for debugging
                        if st.session_state.verbose:
                            st.write("--- Final RAG Output ---")
                            st.write({
                                "question": result.get("question"),
                                "documents": [doc.page_content[:100] + "..." for doc in result.get("documents", [])],
                                "web_search": result.get("web_search", "No"),
                                "generation": response
                            })
                    except Exception as e:
                        st.error(f"Error during RAG processing: {e}")
                        st.error(traceback.format_exc())
                        response = f"Error in RAG processing: {str(e)}"
            except Exception as e:
                st.error(f"Error in RAG outer block: {e}")
                st.error(traceback.format_exc())
                response = f"I encountered an error while processing your request with RAG: {str(e)}"
        elif llm:
            full_response = ""
            message_placeholder = st.empty()
            llm_input_messages = [
                SystemMessage(content="You are a helpful assistant.")
            ] + [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]
            try:
                for chunk in llm.stream(llm_input_messages):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                response = full_response
            except Exception as e:
                st.error(f"LLM streaming failed: {e}")
                response = "I encountered an error while generating a response."
        else:
            response = "LLM not initialized. Please check your API key."

        # Always add the response to messages regardless of how it was generated
        st.session_state.messages.append({"role": "assistant", "content": response})
        return response

    # Display Existing Messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle New User Input
    if user_query := st.chat_input("Enter your message"):
        # Display user message immediately
        user_msg = st.chat_message("user")
        user_msg.write(user_query)
        
        # Create a container for the assistant's response
        with st.chat_message("assistant"):
            # Process the message and get response
            assistant_response = handle_message(user_query)
            # The response will be displayed here
            st.write(assistant_response)

else:
    st.warning("Please provide a valid OpenRouter API Key to continue.")