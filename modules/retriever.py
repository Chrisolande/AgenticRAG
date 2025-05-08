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

def load_documents(docs_dir: str = "books") -> List:
    """
    Load documents from directory
    
    """
    if not os.path.exists(docs_dir):
        raise ValueError(f"The specified directory {docs_dir} does not exist. Please enter a valid directory")

    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def create_vectorstore(documents, embeddings=None, store_type: str = "faiss", persist_directory: Optional[str] = None):
    """
    Create vector store from documents
    
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split the documents into {len(chunks)} chunks")
    
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
    # Create Vector Store
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

def initialize_retriever(docs_dir: str = "books", 
                         store_type: str = "faiss", 
                         persist_directory: Optional[str] = "vector", 
                         similarity_threshold=0.4):
    """
    Initialize retriever

    """
    # Check if vector store exists
    vector_store = None
    if persist_directory and os.path.exists(persist_directory):
        # Load existing vector store
        print(f"Loading vector store from {persist_directory}")
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        if store_type.lower() == "faiss":
            vector_store = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
        elif store_type.lower() == "chroma":
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            
    # If vector store doesn't exist, create it
    if vector_store is None:
        documents = load_documents(docs_dir)
        if not documents:
            print("No documents in the directory")
            return None
        
        # Create Vector Store
        vector_store = create_vectorstore(
            documents,
            store_type=store_type,
            persist_directory=persist_directory
        )

    # Base Retriever
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create the embeddings and the embeddings filter
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=similarity_threshold
    )

    # Create LLMChain Extractor to extract the relevant documents
    llm = initialize_llm()
    llm_extractor = LLMChainExtractor.from_llm(llm=llm)

    # Create a pipeline of compressors
    compression_pipeline = DocumentCompressorPipeline(
        transformers=[embeddings_filter, llm_extractor]
    )

    # Create the retriever
    retriever = ContextualCompressionRetriever(
        base_compressor=compression_pipeline,
        base_retriever=base_retriever
    )

    return retriever
