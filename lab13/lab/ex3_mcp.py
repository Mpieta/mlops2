import datetime
from fastmcp import FastMCP

mcp = FastMCP("Time Service")

@mcp.tool(description="Get the current date in the format YYYY-MM-DD.")
def get_current_date() -> str:

    return datetime.date.today().isoformat()

@mcp.tool(description="Get the current date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS).")
def get_current_datetime() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')

if __name__ == "__main__":
    print("Starting MCP Server on port 8002")
    mcp.run(transport="streamable-http", port=8002)