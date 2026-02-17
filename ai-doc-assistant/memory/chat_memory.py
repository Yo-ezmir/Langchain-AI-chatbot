from langchain.memory import ConversationalBufferMemory

def get_memory():
    return ConversationalBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
