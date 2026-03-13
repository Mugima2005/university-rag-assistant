# university-rag-assistant
# 🎓 AI University Knowledge Assistant (RAG Chatbot)

An AI-powered chatbot that answers questions about university policies, academic regulations, and holiday schedules by retrieving information directly from official documents.

This project uses **Retrieval-Augmented Generation (RAG)** to combine semantic search with a Large Language Model to provide accurate and context-aware answers.

---

## 🚀 Features

- Ask questions about university documents
- Semantic search using vector embeddings
- Context-aware answer generation using an LLM
- Displays answers grounded in official documents
- Interactive chatbot interface using Streamlit

---

## 🧠 Tech Stack

- **Python**
- **SentenceTransformers** – text embeddings
- **Pinecone** – vector database
- **Groq (Llama 3.1)** – LLM for answer generation
- **Streamlit** – web interface
- **PyPDF** – document text extraction

---

## 🏗️ System Architecture
PDF Documents
↓
Text Extraction (PyPDF)
↓
Document Chunking
↓
SentenceTransformer Embeddings
↓
Pinecone Vector Database
↓
User Query
↓
Query Embedding
↓
Vector Similarity Search
↓
Retrieve Relevant Context
↓
Groq LLM (Llama 3.1)
↓
Generated Answer
