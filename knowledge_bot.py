import os
from groq import Groq
import streamlit as st
# from dotenv import load_dotenv

# v0.2 Compatible Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# load_dotenv()

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

# --- Page Setup ---
st.set_page_config(page_title="RAG Campaign Bot", layout="centered")
st.header("🤖 Campaign Knowledge Bot")
st.info("I only answer from the provided documents, brand guidelines and case studies. \n you can ask me about social media marketing, digital marketing and Coca-Cola brand guidelines.")

# --- Configuration ---
PDF_FOLDER = "./documents"
DB_DIR = "./vector_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Step 1: Load & Vectorize Documents ---
@st.cache_resource
def setup_knowledge_base():
    if not os.path.exists(PDF_FOLDER) or not os.listdir(PDF_FOLDER):
        st.error(f"Folder '{PDF_FOLDER}' is empty! Please add atleast 3 PDFs.")
        return None

    documents = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
            documents.extend(loader.load())

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # Embeddings (HuggingFace is free and works great with LangChain)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vector Store
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    return vectorstore

# Initialize
vectorstore = setup_knowledge_base()

# --- Step 2: Chat Logic ---
if vectorstore:
    llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

    # STRICT PROMPT (Requirement: Refuse outside knowledge + Source + Quote)
    system_prompt = (
        "You are a strict Campaign Assistant. Answer ONLY using the provided context. "
        "If the answer is not in the context, say: 'I'm sorry, I can only answer questions based on the provided documents.' "
        "For every answer, you MUST provide the 'Source Document' name and a 'Relevant Quote'."
        "\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Chain setup
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 3}), combine_docs_chain)

    # UI
    user_input = st.text_input("Ask about provided documents, brand guidelines, or case studies:")

    if user_input:
        with st.spinner("Analyzing documents..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            
            # Formatting Output
            st.markdown("### 📝 Answer")
            st.write(answer)
            
            # Showing Source (Requirement: Source Document Name)
            with st.expander("📚 View Sources & Quotes"):
                for doc in response["context"]:
                    st.write(f"**Document:** {doc.metadata['source']}")
                    st.write(f"**Excerpt:** {doc.page_content[:200]}...")
                    st.divider()