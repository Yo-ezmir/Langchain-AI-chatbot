from langchain.vectorstores import Chroma

def create_vectorstore(chunks, embeddings):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
