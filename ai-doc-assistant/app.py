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

st.set_page_config(page_title="AI Document Assistant")

st.title("AI Document Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    # Only rebuild pipeline if new file uploaded
    if (
        "chain" not in st.session_state
        or st.session_state.get("last_file") != uploaded_file.name
    ):

        st.info("Processing document...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        documents = load_pdf(file_path)
        chunks = split_documents(documents)
        embeddings = get_embeddings()
        vectorstore = create_vectorstore(chunks, embeddings)
        memory = get_memory()

        st.session_state.chain = build_chain(vectorstore, memory)
        st.session_state.last_file = uploaded_file.name

        st.success("Document processed successfully!")

    question = st.text_input("Ask a question")

    if question:
        result = st.session_state.chain.invoke({"question": question})
        st.write("### Answer:")
        st.write(result["answer"])
