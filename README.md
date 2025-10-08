# Developing a Custom Chatbot Leveraging Website and Document Data through RAG and LangChain 

- **Author:** Lam Hoang
- **Tech Stack:** Python, RAG, Langchain, Streamlit

## Overview

- **RAG-Chatbot** is a complete end-to-end system that demonstrates how **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)** can be integrated to build an intelligent question-answering chatbot from custom documents and website data.

- The system integrates data crawling, preprocessing, embedding generation, retrieval, and response generation within a unified pipeline.

- At its core, the project uses **LangChain**, **FAISS**, and **LLMs** to deliver context-aware, accurate responses, deployed through an interactive Streamlit web application.

## Key Features

* **Data Collection & Preprocessing:**
  Automatically crawl and clean textual content from websites and documents (PDF, DOCX, etc.) for knowledge base creation.

* **Knowledge Base Construction:**
  Generate embeddings and build a **vector database (FAISS)** to enable efficient information retrieval.

* **RAG Pipeline Implementation:**
  Combine retriever and generator components using LangChain to produce precise and contextually relevant answers.

* **Interactive Streamlit App:**
  User-friendly Streamlit app (`app.py`) that allows querying the chatbot and viewing retrieved contexts in real time.

## Project Structure

```bash
├── src/
│   ├── data_manager   
│   │   ├── process_data.py                 # Preprocess and clean raw text data
│   │   └── web_crawler.py                  # Crawl and extract text content from target websites
│   ├── rag/              
│   │   ├── data_loader/load_data.py        # Load and format data from documents or websites
│   │   ├── vector_manager/vectorstore.py   # Build and manage vector database for retrieval 
│   │   ├── chain.py                        # Define RAG pipeline combining retriever and LLM
│   │   └── create_knowlegde.py             # Create knowledge base and generate embeddings
│   └── app.py                              # Streamlit web app for user interaction
└── README.md
```