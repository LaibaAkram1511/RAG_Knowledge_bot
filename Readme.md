# RAG-Based Agency Knowledge Bot

## Description
A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, and **Groq (Llama-3)**. It allows users to ask questions about agency case studies and brand guidelines provided in PDF format.

## Features
- **Strict Context:** Answers only from the provided documents.
- **Source Attribution:** Each response includes the Source Document Name and a Relevant Quote.
- **Vector Storage:** Uses **ChromaDB** for efficient document retrieval.
- **Embeddings:** Uses HuggingFace's `all-MiniLM-L6-v2` for high-quality local embeddings.

## Local Setup & Run
1. **Prepare Environment:**
   Create a `.env` file in the root directory and add:
   `GROQ_API_KEY=your_api_key_here`

2. **Add Documents:**
   Place exactly 3 PDF files (Case Studies/Guidelines) in the `./documents` folder.

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

## Run the Chatbot
streamlit run knowledge_bot.py