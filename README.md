# AI Agents

Simple AI agent projects built using **LangGraph, LangChain, and Groq**.



### Drafter Agent

A document editing agent that can:

* Create and modify documents
* Update document content
* Save documents as `.txt` files

### RAG Agent

A RAG-based agent that:

* Loads a PDF document
* Creates embeddings using `BAAI/bge-m3`
* Stores embeddings in ChromaDB
* Retrieves relevant information to answer questions

## Tech Stack

* Python
* LangGraph
* LangChain
* Groq
* ChromaDB
* Hugging Face

## Setup

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

Install dependencies:

```bash
uv sync
```

Run an agent from its respective folder:

```bash
uv run main.py
```
