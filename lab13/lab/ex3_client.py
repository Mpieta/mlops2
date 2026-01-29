import asyncio
import json
import os
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()


class MCPManager:
    def __init__(self, servers: dict[str, str]):
        self.servers = servers
        self.clients = {}
        self.tools = []
        self._stack = AsyncExitStack()

    async def __aenter__(self):
        for url in self.servers.values():
            read, write, _ = await self._stack.enter_async_context(
                streamable_http_client(url)
            )
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
        "weather": "http://localhost:8001/mcp",
        "time": "http://localhost:8002/mcp",
    }

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    model_name = "Qwen/Qwen3-1.7B"


    async with MCPManager(mcp_servers) as mcp:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use tools for date and weather info."},
            {"role": "user", "content": prompt},
        ]

        for _ in range(10):
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

                print(f"Executing MCP Tool: {name}")
                result = await mcp.call_tool(name, args)
                print(f"Tool result: {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": str(result),
                })


if __name__ == "__main__":
    test_prompt = "What will the weather be in London exactly two weeks from now?"
    print(asyncio.run(make_llm_request(test_prompt)))