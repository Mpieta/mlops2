import os
from typing import Annotated
from fastmcp import FastMCP
from tavily import TavilyClient

api_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=api_key)
mcp = FastMCP("Tavily Search Service")


@mcp.tool(description="Search the web for travel information, attractions, and local tips.")
def web_search(
        query: Annotated[str, "The search query"]
) -> str:
    try:
        response = tavily.search(query=query, search_depth="basic")
        results = response.get("results", [])

        summary = ["Search Results:"]
        for res in results[:3]:
            summary.append(f"- {res['title']}: {res['content'][:200]}... ({res['url']})")

        return "\n".join(summary)
    except Exception as e:
        return f"Search error: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8003)