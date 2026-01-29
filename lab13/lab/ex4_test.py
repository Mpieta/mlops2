import asyncio
import json
import base64
from contextlib import AsyncExitStack
from openai import OpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPManager:
    def __init__(self, servers: dict[str, str]):
        self.servers = servers
        self.clients = {}
        self.tools = []
        self._stack = AsyncExitStack()

    async def __aenter__(self):
        for name, url in self.servers.items():
            read, write, _ = await self._stack.enter_async_context(streamable_http_client(url))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools_resp = await session.list_tools()
            for t in tools_resp.tools:
                self.clients[t.name] = session
                self.tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                })
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stack.aclose()

    async def call_tool(self, name: str, args: dict) -> str:
        result = await self.clients[name].call_tool(name, arguments=args)
        return result.content[0].text


async def make_llm_request(prompt: str) -> str:
    mcp_servers = {
        "viz_server": "http://localhost:8003/mcp"
    }

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    model_name = "Qwen/Qwen3-1.7B"

    async with MCPManager(mcp_servers) as mcp:
        messages = [
            {"role": "system",
             "content": "You are a data assistant. Use the line_plot tool to visualize numerical data when asked."},
            {"role": "user", "content": prompt},
        ]

        for _ in range(5):
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=mcp.tools,
                tool_choice="auto",
            )

            resp_msg = response.choices[0].message
            if not resp_msg.tool_calls:
                return resp_msg.content

            messages.append(resp_msg)
            for tool_call in resp_msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                print(f"Using tool: {name}")
                print(f"Tool arguments: {args}")

                result_b64 = await mcp.call_tool(name, args)

                if name == "line_plot":
                    filename = "generated_plot.png"
                    with open(filename, "wb") as f:
                        f.write(base64.b64decode(result_b64))

                    tool_response_text = f"[Image saved as {filename}]"
                else:
                    tool_response_text = result_b64

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": tool_response_text,
                })


if __name__ == "__main__":
    prompt = "I have these sales figures for the week: 120, 150, 110, 200, 180, 210, 250. Can you create a line plot for me with the title 'Weekly Sales'?"
    print(prompt)
    print("Response:\n")
    final_text = asyncio.run(make_llm_request(prompt))
    print(final_text)