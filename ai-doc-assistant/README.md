# AI Document Assistant (RAG-Chatbot)

A modular Retrieval-Augmented Generation (RAG) application built with LangChain and Streamlit. 
This assistant allows users to upload PDF documents and engage in a context-aware 
conversation about the content using OpenAI LLMs.

## Features
- PDF Ingestion: Load and process complex PDF documents.
- Intelligent Chunking: Efficiently splits text to maintain context for the LLM.
- Vector Search: Uses Retrievers to find specific data from documents.
- Conversational Memory: Retains chat history for follow-up questions.
- Agentic Capabilities: Integrated Tools for dynamic reasoning and external function calls.

## Project Structure
ai-doc-assistant
├── app.py        - Streamlit Frontend U                
├── loaders/      -Document loading logic (PyPDF)             
├── processing/     -Text splitting and cleaning           
├── embeddings/      -Vector embedding configurations        
├── vectorstore/      -ChromaDB management        
├── memory/            -Chat history and BufferMemory       
├── chains/            -ConversationalRetrievalChain setup        
├── tools/             -Custom Agent tools       
└── requirements.txt    Project dependencies       

## Building Blocks
Following the principles of Building Intelligent LLM Apps, this project implements:
1. The Brains (LLM): GPT-4o-mini via langchain-openai.
2. Retrievers: Finding data from documents to provide context.
3. Memory: Tracking past interactions for context retention.
4. Chains: Connecting prompts and logic steps into a multi-step workflow.
5. Agents: Letting the AI choose tools dynamically based on the query.

## Installation and Setup
Run the following commands in your terminal to set up the project:

git clone https://github.com/your-username/ai-doc-assistant.git
cd ai-doc-assistant
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

## Environment Variables
Create a .env file in the root directory and add your API key:
OPENAI_API_KEY=your_actual_key_here

## Usage
Run the application using the Streamlit module flag:
python -m streamlit run app.py

Developer: Yonatan Azmir
Date: February 2026
