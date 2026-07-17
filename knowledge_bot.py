import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# ---------------- PAGE SETUP ----------------

st.set_page_config(
    page_title="RAG Campaign Bot",
    layout="centered"
)

st.header("🤖 Campaign Knowledge Bot")

st.info(
    """
I only answer from the provided documents, brand guidelines and case studies.

You can ask about social media marketing, digital marketing and brand guidelines.
"""
)


# ---------------- CONFIGURATION ----------------

PDF_FOLDER = "./documents"
DB_DIR = "./vector_db"

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# ---------------- LOAD DOCUMENTS ----------------

@st.cache_resource
def setup_knowledge_base():

    if not os.path.exists(PDF_FOLDER) or not os.listdir(PDF_FOLDER):

        st.error(
            f"Folder '{PDF_FOLDER}' is empty! Please add PDFs."
        )

        return None


    documents = []


    for file in os.listdir(PDF_FOLDER):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(
                os.path.join(PDF_FOLDER, file)
            )

            documents.extend(
                loader.load()
            )


    if not documents:

        st.error(
            "No PDF documents found."
        )

        return None


    # Text Chunking

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=100

    )


    splits = text_splitter.split_documents(
        documents
    )


    # Embeddings

    embeddings = HuggingFaceEmbeddings(

        model_name="all-MiniLM-L6-v2"

    )


    # Vector Database

    vectorstore = Chroma.from_documents(

        documents=splits,

        embedding=embeddings,

        persist_directory=DB_DIR

    )


    return vectorstore



# Initialize Knowledge Base

vectorstore = setup_knowledge_base()



# ---------------- RAG CHAT ----------------

if vectorstore:


    llm = ChatGroq(

        temperature=0,

        model_name="llama-3.1-8b-instant",

        groq_api_key=GROQ_API_KEY

    )


    system_prompt = (

        "You are a strict Campaign Assistant. "

        "Answer ONLY using the provided context. "

        "If the answer is not available in context, say: "

        "'I'm sorry, I can only answer questions based on the provided documents.' "

        "Always provide Source Document name and Relevant Quote."

        "\n\n"

        "Context: {context}"

    )


    prompt = ChatPromptTemplate.from_messages(

        [

            ("system", system_prompt),

            ("human", "{input}")

        ]

    )



    combine_docs_chain = create_stuff_documents_chain(

        llm,

        prompt

    )


    rag_chain = create_retrieval_chain(

        vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),

        combine_docs_chain

    )



    # ---------------- UI ----------------


    user_input = st.text_input(

        "Ask about provided documents, brand guidelines, or case studies:"

    )


    if user_input:


        with st.spinner(
            "Analyzing documents..."
        ):


            response = rag_chain.invoke(

                {
                    "input": user_input
                }

            )


            answer = response["answer"]



        st.markdown(
            "### 📝 Answer"
        )


        st.write(
            answer
        )



        with st.expander(
            "📚 View Sources & Quotes"
        ):


            for doc in response["context"]:


                st.write(
                    f"**Document:** {doc.metadata.get('source','Unknown')}"
                )


                st.write(
                    f"**Excerpt:** {doc.page_content[:200]}..."
                )


                st.divider()