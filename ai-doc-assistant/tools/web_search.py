from duckduckgo_search import DDGS
from langchain.tools import Tool

def search_web(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)

        output = ""
        for r in results:
            output += r["title"] + "\n"
            output += r["body"] + "\n\n"

        return output

web_search_tool = Tool(
    name="Web Search",
    func=search_web,
    description="Search the web for information not found in the document"
)
