import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

st.title("Groq AI – Research Paper Q&A")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    pdf_reader = PdfReader(pdf)
    text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

    st.success("PDF processed successfully.")

query = st.text_input("Ask anything about your PDF")

if query and st.session_state.vector_store:
    docs = st.session_state.vector_store.similarity_search(query, k=4)

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)

    st.success(answer)