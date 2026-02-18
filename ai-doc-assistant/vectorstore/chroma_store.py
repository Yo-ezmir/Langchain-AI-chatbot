from langchain_community.vectorstores import Chroma

def create_vectorstore(chunks, embeddings):
    """
    Creates a persistent Chroma database.
    Note: 'persist_directory' now automatically handles saving; 
    there is no need to call db.persist() separately.
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="document_assistant" # Good practice to name your collection
    )
    return vectorstore


