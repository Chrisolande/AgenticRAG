# 📚 AgenticRAG

> *Unlock the wisdom of texts through intelligent retrieval and generation*

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![Status](https://img.shields.io/badge/status-active-success.svg)

## ✨ What is AgenticRAG?

AgenticRAG is an intelligent system that combines the power of retrieval with generative AI to provide accurate, contextually-rich responses from literary and historical texts. Think of it as having a scholar who has read thousands of books and can instantly find and synthesize relevant information to answer your questions.

## 🔍 How It Works

```
📄 Query → 🔎 Semantic Search → 📑 Retrieval → 🧠 Context Analysis → ✍️ Response Generation
```

1. **Your question** is processed to understand its intent
2. **Semantic search** finds the most relevant passages from our corpus
3. **Retrieved content** provides factual grounding
4. **AI generation** crafts a coherent, accurate response

## 🧠 Architecture

AgenticRAG uses a sophisticated architecture with multiple specialized components:

- **Question Router** - Determines whether to use retrieval, web search, or direct generation
- **Document Retrieval** - Finds relevant passages from the corpus
- **Web Search** - Integrates external knowledge when needed
- **Contextual Compression** - Optimizes retrieved content by removing irrelevant information
- **Retrieval Grader** - Evaluates the quality and relevance of retrieved documents
- **RAG Generation Chain** - Combines retrieved context with generative capabilities
- **Final Response** - Delivers accurate, contextual answers

## 🌟 Key Features

- **Rich Text Corpus** - Classic literature and historical documents at your fingertips
- **Intelligent Retrieval** - Advanced semantic search using vector embeddings
- **Knowledge-Grounded Responses** - Reduces hallucinations with fact-based generation
- **High Performance** - Optimized with FAISS for fast, scalable search
- **Modular Architecture** - Easily extensible for custom applications

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Langchain 0.3.25+
- Langhraph 0.4.3+
- Langchain openai 0.3.16+
- Langchain core 0.3.59
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy .env.example to .env and add your API keys
cp .env.example .env

# Run the application
python modules/main.py
```

## 📋 Usage

### Command Line Interface

```bash
# Interactive mode (default)
python modules/main.py

# Single query mode
python modules/main.py --query "What are the major themes in Frankenstein?"

# Verbose mode to see intermediate processing steps
python modules/main.py --verbose

# Explicit interactive mode with verbose output
python modules/main.py --interactive --verbose

# View all available options
python modules/main.py --help
```

AgenticRAG can also be used through:

- **Python API** - Integration with other applications
- **Interactive notebook** - Explore with visualizations

## 📁 Project Structure



```
📦 agentic-rag
 ┣ 📂 books                  # Text corpus collection
 ┣ 📂 vector                 # Pre-computed vector indices
 ┣ 📂 modules                # Core functionality
 ┃ ┣ 📜 chains.py            # RAG workflow orchestration
 ┃ ┣ 📜 graph.py             # Knowledge graph operations
 ┃ ┣ 📜 main.py              # Application entry point
 ┃ ┣ 📜 prompts.py           # Prompt engineering utilities
 ┃ ┣ 📜 retriever.py         # Semantic search implementation
 ┃ ┗ 📜 utils.py             # Helper functions
 ┣ 📓 Agentic RAG.ipynb      # Interactive demo notebook
 ┣ 📜 .env                   # Environment variables
 ┣ 📜 .gitignore
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

## 📚 Learn More About RAG

Retrieval-Augmented Generation (RAG) enhances AI language models by:

- **Grounding responses** in verified information
- **Incorporating new knowledge** without retraining
- **Enabling source citation** for transparency
- **Reducing computational requirements** compared to larger models

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  Made with ❤️ for the intersection of literature and artificial intelligence
</p>