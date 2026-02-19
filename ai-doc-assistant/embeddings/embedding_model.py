from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings(provider="openai", api_key=None):
    """Get embedding model based on provider. Falls back to OpenAI."""
    if provider == "google":
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
    # OpenAI and Anthropic both use OpenAI embeddings
    # (Anthropic doesn't have its own embedding model)
    return OpenAIEmbeddings(api_key=api_key)
