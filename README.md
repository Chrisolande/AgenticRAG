# AgenticRAG
================

**Retrieval-Augmented Generation System for Text Analysis and Response**

## Overview
AgenticRAG is a Retrieval-Augmented Generation (RAG) system designed to analyze and respond to queries using a corpus of literary and historical texts. By combining semantic search with generative AI, it provides accurate and contextually relevant answers. This project is ideal for exploring the intersection of natural language processing, information retrieval, and knowledge generation.

### What is RAG?
Retrieval-Augmented Generation (RAG) is a technique that enhances generative AI models by integrating them with a retrieval system. Instead of relying solely on pre-trained knowledge, RAG retrieves relevant information from an external corpus to ground its responses, improving accuracy and reducing hallucinations.

## Key Features
- **Text Corpus**: A rich collection of classic literature and historical documents located in the `books/` directory.
- **Semantic Search**: Efficient retrieval of relevant text passages using vector embeddings and FAISS indexing.
- **Generative AI**: Combines retrieved information with generative models to produce coherent and informed responses.
- **Modular Design**: Flexible architecture with dedicated modules for chains, graph operations, prompting, retrieval, and utilities.

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

### Example Query
After running the application, you can input queries like:
- "Summarize the plot of 'Pride and Prejudice'."
- "What are the key themes in 'War and Peace'?"
- "Provide a comparison between 'The Iliad' and 'The Odyssey'."

## Contributing
Pull requests are welcome! Please ensure that any additions are well-documented and include relevant tests.

## References
- [LangChain Documentation](https://langchain.readthedocs.io/)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki/Tutorial)
- [Transformers Library](https://huggingface.co/docs/transformers/)
