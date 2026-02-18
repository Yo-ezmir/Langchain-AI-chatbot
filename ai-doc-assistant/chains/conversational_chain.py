
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain # type: ignore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain # type: ignore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def build_chain(vectorstore):
    # 1. Setup the Brains (LLM)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # 2. Setup the Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Create a History-Aware Retriever
    # This block "rewrites" the user's question to be standalone using chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 4. Create the Question-Answering Chain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 5. Build the Final RAG Chain
    # Note: Memory is now handled in the 'app.py' via chat_history list
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)