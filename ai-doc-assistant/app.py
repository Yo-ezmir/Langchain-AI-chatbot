import streamlit as st
import tempfile
from dotenv import load_dotenv

from loaders.document_loader import load_pdf
from processing.text_splitter import split_documents
from embeddings.embedding_model import get_embeddings
from vectorstore.chroma_store import create_vectorstore
from memory.chat_memory import get_memory
from chains.conversational_chain import build_chain

load_dotenv()

st.title("AI Document Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)
    memory = get_memory()
    qa_chain = build_chain(vectorstore, memory)

    question = st.text_input("Ask a question")

    if question:
        result = qa_chain.invoke({"question": question})
        st.write(result["answer"])
