# Endee Documentation AI Assistant

This project demonstrates a Retrieval-Augmented Generation (RAG) system built using the Endee vector database.

The application allows users to ask questions about Endee documentation and receive AI-generated answers based on retrieved context.

---

## Project Overview

This project combines:

- Endee Vector Database
- Sentence Transformers Embedding Model
- Retrieval-Augmented Generation (RAG)
- Groq LLM API
- Streamlit Web Interface

The assistant retrieves relevant documentation from a vector database and uses a large language model to generate answers.

---

## System Architecture

User Question  
↓  
Embedding Model (Sentence Transformers)  
↓  
Endee Vector Database  
↓  
Top-K Context Retrieval  
↓  
LLM (Groq / Llama3)  
↓  
Generated Answer  
↓  
Streamlit Web UI

---

## Features

- Semantic search over Endee documentation
- Vector similarity search using Endee
- Retrieval-Augmented Generation pipeline
- Interactive web interface with Streamlit
- Docker and API documentation retrieval

---

## Technologies Used

- Python
- Endee Vector Database
- Sentence Transformers
- Streamlit
- Groq LLM API

---

## Setup Instructions

### 1. Clone the repository
https://github.com/kumarc101203/endee
cd endee/examples/knowledge_rag


### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate


### 3. Install dependencies
pip install -r requirements.txt


### 4. Start Endee server
docker run --ulimit nofile=100000:100000 -p 8080:8080 -v ./endee-data:/data endeeio/endee-server:latest


### 5. Ingest documentation
python ingest.py


### 6. Run AI assistant
streamlit run app.py


---

## Example Queries

- How does Endee run with Docker?
- What is a vector database?
- How do you create an index in Endee?
- How does vector similarity search work?

---

## Use Cases

- Documentation assistants
- Semantic search systems
- AI chatbots over knowledge bases
- Retrieval-Augmented Generation pipelines