import os
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter, 
    LLMChainFilter, 
    LLMChainExtractor, 
    DocumentCompressorPipeline
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from utils import initialize_llm

# Function to load documents from a specified directory
def load_documents(docs_dir: str = "books") -> List:
    """
    Load text documents from the specified directory.
    Args:
        docs_dir (str): Path to the directory containing text files.
    Returns:
        List: A list of loaded documents.
    """
    if not os.path.exists(docs_dir):
        raise ValueError(f"The specified directory {docs_dir} does not exist. Please enter a valid directory")

    # Load all .txt files from the directory
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

# Function to create a vector store from documents
def create_vectorstore(documents, embeddings=None, store_type: str = "faiss", persist_directory: Optional[str] = None):
    """
    Create a vector store for efficient document retrieval.
    Args:
        documents (List): List of documents to index.
        embeddings: Embedding model to use for vectorization.
        store_type (str): Type of vector store (e.g., 'faiss', 'chroma').
        persist_directory (Optional[str]): Directory to save the vector store.
    Returns:
        VectorStore: The created vector store.
    """
    # Split documents into smaller chunks for indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split the documents into {len(chunks)} chunks")
    
    # Use default embeddings if none are provided
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
    # Create the vector store based on the specified type
    if store_type.lower() == "faiss":
        vector_store = FAISS.from_documents(chunks, embeddings)
        if persist_directory:
            vector_store.save_local(persist_directory)
    elif store_type.lower() == "chroma":
        if persist_directory:            
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            vector_store.persist()
        else:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )
    else:
        raise ValueError(f"Unknown vector store type {store_type}")

    return vector_store

# Function to initialize a retriever with contextual compression
def initialize_retriever(docs_dir: str = "books", 
                         store_type: str = "faiss", 
                         persist_directory: Optional[str] = "vector", 
                         similarity_threshold=0.4):
    """
    Initialize a retriever for document search and retrieval.
    Args:
        docs_dir (str): Path to the directory containing documents.
        store_type (str): Type of vector store (e.g., 'faiss', 'chroma').
        persist_directory (Optional[str]): Directory to save/load the vector store.
        similarity_threshold (float): Threshold for filtering similar embeddings.
    Returns:
        ContextualCompressionRetriever: The initialized retriever.
    """
    # Check if a pre-existing vector store exists
    vector_store = None
    if persist_directory and os.path.exists(persist_directory):
        print(f"Loading vector store from {persist_directory}")
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        if store_type.lower() == "faiss":
            vector_store = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
        elif store_type.lower() == "chroma":
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            
    # If no vector store exists, create a new one
    if vector_store is None:
        documents = load_documents(docs_dir)
        if not documents:
            print("No documents in the directory")
            return None
        
        vector_store = create_vectorstore(
            documents,
            store_type=store_type,
            persist_directory=persist_directory
        )

    # Create a base retriever for similarity search
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create an embeddings filter for similarity-based compression
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=similarity_threshold
    )

    # Create an LLM-based extractor for relevant documents
    llm = initialize_llm()
    llm_extractor = LLMChainExtractor.from_llm(llm=llm)

    # Combine filters and extractors into a compression pipeline
    compression_pipeline = DocumentCompressorPipeline(
        transformers=[embeddings_filter, llm_extractor]
    )

    # Create the final retriever with contextual compression
    retriever = ContextualCompressionRetriever(
        base_compressor=compression_pipeline,
        base_retriever=base_retriever
    )

    return retriever
