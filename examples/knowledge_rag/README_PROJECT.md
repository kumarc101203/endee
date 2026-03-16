# Endee RAG Knowledge Assistant

## Project Overview
This project demonstrates a Retrieval-Augmented Generation (RAG) system built using the Endee vector database.

The system allows users to upload documents and ask questions about them. Relevant document chunks are retrieved using vector similarity search in Endee.

## Architecture

Documents
↓
Embedding Model (Sentence Transformers)
↓
Endee Vector Database
↓
Similarity Search
↓
LLM Response

## Environment Setup

### 1. Fork Repository
Forked from:
https://github.com/endee-io/endee

### 2. Start Endee Server

docker run \
--ulimit nofile=100000:100000 \
-p 8080:8080 \
-v ./endee-data:/data \
--name endee-server \
--restart unless-stopped \
endeeio/endee-server:latest

Server runs at:
http://localhost:8080

### 3. Python Environment

python -m venv venv
venv\Scripts\activate

### 4. Install Dependencies

pip install -r requirements.txt

## Current Progress

✔ Endee server running  
✔ Python environment configured  
✔ RAG project structure created  

## Next Steps

- Implement document ingestion pipeline
- Generate embeddings
- Insert vectors into Endee
- Build retrieval interface