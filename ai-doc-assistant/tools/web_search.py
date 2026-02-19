from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo and return formatted results."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)

            if not results:
                return "No web results found."

            output = ""
            for r in results:
                output += f"**{r['title']}**\n"
                output += f"{r['body']}\n"
                output += f"[Source]({r['href']})\n\n"

            return output
    except Exception as e:
        return f"Web search failed: {str(e)}"
