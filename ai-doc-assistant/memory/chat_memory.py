class ConversationBufferMemory:
    def __init__(self, memory_key="chat_history", return_messages=True):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.messages = []

    def save_context(self, inputs, outputs):
        if "question" in inputs:
            self.messages.append({"role": "user", "content": inputs["question"]})
        if "answer" in outputs:
            self.messages.append({"role": "ai", "content": outputs["answer"]})

    def load_memory_variables(self, inputs):
        if self.return_messages:
            return {self.memory_key: self.messages}
        return {self.memory_key: [m["content"] for m in self.messages]}

def get_memory():
    return ConversationBufferMemory()

"""from langchain.memory import ConversationBufferMemory

def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
"""