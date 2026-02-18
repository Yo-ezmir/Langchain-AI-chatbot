from langchain_classic.memory import ConversationBufferMemory # type: ignore

def get_memory():
    """
    Returns a standard LangChain memory buffer.
    'chat_history' is the key the chain looks for.
    'return_messages=True' ensures it returns objects, not just strings.
    """
    return ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
