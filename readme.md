# Document Buddy App

A powerful Streamlit application to interact with documents using AI. Upload PDF documents, create embeddings, and chat with your document to get answers to your questions.

## Features
- Upload PDF documents.
- Create and store embeddings using Qdrant.
- Chat with an AI chatbot that retrieves and generates answers from your document.
- Uses HuggingFace and Qdrant for embeddings and vector database storage.
- Supports real-time streaming responses.

## Prerequisites
- Python 3.9+
- Qdrant server running locally (default at `http://localhost:6333`).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/document-buddy.git
   cd document-buddy
2. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows, use `.venv\Scripts\activate`
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Start the Qdrant server:
   ```bash
   docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
5. Run the application:
   ```bash
   streamlit run app.py
