from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain

def build_chain(vectorstore, memory):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
