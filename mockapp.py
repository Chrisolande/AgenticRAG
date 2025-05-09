import streamlit as st
import uuid
import os
import dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

dotenv.load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

MODELS = [
    "google/gemini-2.5-pro-exp-03-25",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"
]

st.set_page_config(
    page_title="RAG LLM app?", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- Header ---

st.html("""<h2 style="text-align: center;">📚🔍 <i> Literature Agentic RAG </i> 🤖💬</h2>""")

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state['rag_sources'] = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!, How may I help you today?"}
    ]

# ---Sidebar Items ---
with st.sidebar:
    if api_key not in os.environ:
        default_api_key = os.getenv("OPENROUTER_API_KEY") if os.getenv("OPENROUTER_API_KEY") is not None else ""
        with st.popover("🔐 OpenRouter"):
            openrouter_api_key = st.text_input(
            "Please Enter your openrouter API Key",
            value = default_api_key,
            type = "password",
            key = "openrouter_api_key")
    else:
        openrouter_api_key = os.environ["OPENROUTER_API_KEY"]
        st.session_state.openrouter_api_key = openrouter_api_key
    
    # --- Main Content ---
    # Has the user provided an api key?
    missing_openrouter = openrouter_api_key =="" or openrouter_api_key is None or "sk-" not in openrouter_api_key
    if missing_openrouter and ("OPENROUTER_API_KEY" not in os.environ):
        st.write("#")
        st.warning("Please introduce an API Key to continue...")
    else:
        # sidebar
        with st.sidebar:
            st.divider()
            available_models = []
            for model in MODELS:
                if "gemini" in model and not missing_openrouter:
                    available_models.append(model)
                elif "mistral" in model and not missing_openrouter:
                    available_models.append(model)
                elif "llama" in model and not missing_openrouter:
                    available_models.append(model)

            st.selectbox(
                "select a model",
                options = available_models,
                key = 'model'
            )
        
        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG",
                value = is_vector_db_loaded,
                key = "use_rag",
                disabled=not is_vector_db_loaded
            )
        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type = "primary")
        
        st.header("RAG resources")

        # File upload
        st.file_uploader(
            "Upload a document",
            type = 'txt',
            accept_multiple_files=True,
            #on_change=load_doc_to_db,
            key = "rag_docs"
        )

        with st.expander(f"Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])
    
    # Main Chat app
    model_provider = st.session_state.model.split("/")[0]
    if "gemini" in available_models:
        llm_stream = ChatOpenAI(
            model_name = st.session_state.model.split("/")[-1],
            api_key = openrouter_api_key,
            temperature = 0.3,
            streaming = True,
            openai_api_base="https://openrouter.ai/api/v1"
        )
    elif "mistral" in available_models:
        llm_stream = ChatOpenAI(
            model_name = st.session_state.model.split("/")[-1],
            api_key = openrouter_api_key,
            temperature = 0.3,
            streaming = True,
            openai_api_base="https://openrouter.ai/api/v1"
        )
    else:
        llm_stream = ChatOpenAI(
            model_name = st.session_state.model.split("/")[-1],
            api_key = openrouter_api_key,
            temperature = 0.3,
            streaming = True,
            openai_api_base="https://openrouter.ai/api/v1"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your Message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Prepare messages for the LLM in chronological order
            llm_input_messages = [
                SystemMessage(content="You are a helpful assistant.")
            ] + [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) 
                for m in st.session_state.messages # Iterate through the flat list of messages
            ]

            for response in llm_stream.stream(llm_input_messages):
                full_response += response.content
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})