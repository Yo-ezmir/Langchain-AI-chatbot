import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from loaders.document_loader import load_pdf
from processing.text_splitter import split_documents
from embeddings.embedding_model import get_embeddings
from vectorstore.chroma_store import create_vectorstore
# Memory is now handled directly in st.session_state, no need for get_memory()
from chains.conversational_chain import build_chain

load_dotenv()

st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("🤖 AI Document Assistant")

# Initialize chat history for the UI and the Chain
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Only process if it's a new file
    if "chain" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

            # Your modular building blocks
            documents = load_pdf(file_path)
            chunks = split_documents(documents)
            embeddings = get_embeddings()
            vectorstore = create_vectorstore(chunks, embeddings)
            
            # Updated: build_chain no longer takes 'memory' as an argument
            st.session_state.chain = build_chain(vectorstore)
            st.session_state.last_file = uploaded_file.name
            st.session_state.chat_history = [] # Reset history for new file
            st.success("Ready to chat!")

# Display existing chat history
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Handle user input
if question := st.chat_input("Ask a question about your document..."):
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Generate response
    if "chain" in st.session_state:
        with st.spinner("Thinking..."):
            # The chain expects 'input' and 'chat_history'
            response = st.session_state.chain.invoke({
                "input": question,
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            
            # 3. Update and display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)
            
            # 4. Append to history (This keeps the conversation going)
            st.session_state.chat_history.append(HumanMessage(content=question))
            st.session_state.chat_history.append(AIMessage(content=answer))
    else:
        st.warning("Please upload a PDF first.")



"""
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
"""