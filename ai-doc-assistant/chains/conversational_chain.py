from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


SYSTEM_PROMPT = """\
You are an intelligent document assistant. Answer the user's question using ONLY \
the provided context from the uploaded document. If the answer is not in the context, \
say so clearly. Be concise. Use bullet points or structured formatting when appropriate.

Context:
{context}
"""


def get_llm(provider="openai", model=None, api_key=None):
    """Create an LLM instance for the given provider."""
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model or "claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.2,
        )
    else:  # openai
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            temperature=0.2,
            streaming=True,
        )


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_chain(vectorstore, provider="openai", model=None, api_key=None):
    """Build a modern LCEL retrieval chain with chat history support."""
    llm = get_llm(provider, model, api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "chat_history": lambda x: x.get("chat_history", []),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
