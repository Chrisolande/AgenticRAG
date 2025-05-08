# AgenticRAG
================

**Retrieval-Augmented Generation System for Text Analysis and Response**

## Overview
AgenticRAG is a project integrating retrieval-augmented generation (RAG) capabilities with a corpus of literary and historical texts. It leverages vector embeddings for efficient semantic search and response generation. Key components include:

- **Text Corpus**: A collection of classic literature and historical documents in `books/`.
- **Vector Index**: Pre-built FAISS index for semantic search (`vector/index.faiss` and `vector/index.pkl`).
- **Modular Architecture**: Python modules for chains, graph operations, prompting, retrieval, and utilities.

## Project Structure
```plain
├── books/                  # Text corpus
├── vector/                 # Pre-computed vector indices
├── modules/                # Core functionality
│   ├── chains.py           # RAG workflow management
│   ├── graph.py            # Knowledge graph interactions
│   ├── main.py             # Entry point for the application
│   ├── prompts.py          # Prompt engineering utilities
│   ├── retriever.py        # Semantic search logic
│   └── utils.py            # Shared helper functions
├── Agentic RAG.ipynb       # Demo/experimental notebook
├── .gitignore
├── .env                    # Environment variables (e.g., API keys)
└── README.md
```

## Dependencies
- `langchain` for RAG pipelines
- `faiss-cpu` (or `faiss-gpu` for accelerated search)
- `transformers` for generation models

## Usage
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt  # (Generate if missing: pip freeze > requirements.txt)
   ```
2. **Rebuild Vector Index (if modified)**:
   ```bash
   python modules/retriever.py --rebuild-index
   ```
3. **Run the Application**:
   ```bash
   python modules/main.py
   ```
4. **Interactive Demo**:
   - Open `Agentic RAG.ipynb` for example workflows

## Contributing
Pull requests welcome! Ensure additions are documented and tested.

## References
- [LangChain Documentation](https://langchain.readthedocs.io/)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki/Tutorial)
