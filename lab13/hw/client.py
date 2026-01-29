import os
import json
import asyncio
from contextlib import AsyncExitStack

from openai import OpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from guardrails import Guard, OnFailAction
from guardrails.hub import DetectJailbreak, RestrictToTopic

guard = Guard() \
    .use(
    DetectJailbreak,
    on="prompt",
    on_fail=OnFailAction.EXCEPTION
) \
    .use(
    RestrictToTopic(
        valid_topics=["travel", "geography", "weather", "hotels", "flights", "tourism", "packing", "budget"],
        invalid_topics=["politics", "medical advice", "illegal acts", "programming", "relationship advice"],
        disable_classifier=False,
        disable_llm=True
    ),
    on="output",
    on_fail=OnFailAction.EXCEPTION
)


class AgentManager:
    def __init__(self):
        self.stack = AsyncExitStack()
        self.mcp_clients = {}
        self.tools = []

        self.llm_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url="http://localhost:8000/v1"
        )

    async def initialize_mcp(self):
        servers = {
            "weather": "http://127.0.0.1:8004/mcp",
            "tavily": "http://127.0.0.1:8003/mcp"
        }

        for name, url in servers.items():
            try:
                read, write, _ = await self.stack.enter_async_context(streamable_http_client(url))
                session = await self.stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                tools_resp = await session.list_tools()
                for t in tools_resp.tools:
                    self.mcp_clients[t.name] = session
                    self.tools.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.inputSchema,
                        }
                    })
                print(f"Connected to {name.upper()} MCP ({len(tools_resp.tools)} tools)")
            except Exception as e:
                print(f"Failed to connect to {name} MCP: {e}")

    async def call_tool(self, name, args):
        if name in self.mcp_clients:
            res = await self.mcp_clients[name].call_tool(name, arguments=args)
            return res.content[0].text
        return "Tool not found."

    async def run_chat_loop(self):
        history = [
            {"role": "system",
             "content": "You are an experienced travel agent. Help the user plan their trip using available tools. Keep responses concise and practical."}
        ]

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break

            try:
                guard.validate(user_input)
            except Exception as e:
                print(f"Request blocked")
                continue

            history.append({"role": "user", "content": user_input})

            try:
                for _ in range(5):
                    response = self.llm_client.chat.completions.create(
                        model="Qwen/Qwen3-1.7B",
                        messages=history,
                        tools=self.tools,
                        tool_choice="auto",
                        max_completion_tokens=1024,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    )

                    msg = response.choices[0].message
                    history.append(msg)

                    if not msg.tool_calls:
                        try:
                            guard.validate(msg.content, metadata={"prompt": user_input})
                        except Exception as e:
                            print(f"\nAgent: Request blocked\n")
                        break

                    # Handle Tools
                    for tool in msg.tool_calls:
                        fname = tool.function.name
                        fargs = json.loads(tool.function.arguments)
                        print(f"\nCalling {fname} with args {fargs}\n")

                        result = await self.call_tool(fname, fargs)
                        print(result)
                        history.append({
                            "role": "tool",
                            "tool_call_id": tool.id,
                            "name": fname,
                            "content": str(result)
                        })

            except Exception as e:
                print(f"Error: {e}")

    async def cleanup(self):
        await self.stack.aclose()


async def main():
    agent = AgentManager()
    try:
        await agent.initialize_mcp()
        await agent.run_chat_loop()
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())