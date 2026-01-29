import datetime
import json
import polars as pl
from typing import Callable
from openai import OpenAI


def make_llm_request(prompt: str) -> str:
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

    messages = [
        {
            "role": "developer",
            "content": "You are a data assistant. When you use a tool, the data will be provided to you in the next message. Use that data to answer the user accurately."
        },
        {"role": "user", "content": prompt},
    ]

    tool_definitions, tool_name_to_func = get_tool_definitions()

    for _ in range(10):
        response = client.chat.completions.create(
            model="Qwen/Qwen3-1.7B",
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
            max_completion_tokens=1000,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        resp_message = response.choices[0].message
        messages.append(resp_message.model_dump())

        if resp_message.tool_calls:
            for tool_call in resp_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)


                func = tool_name_to_func[func_name]
                try:
                    func_result = func(**func_args)
                except Exception as e:
                    func_result = f"Error: {str(e)}"

                messages.append({
                    "role": "tool",
                    "content": json.dumps(func_result),
                    "tool_call_id": tool_call.id,
                })
        else:
            return resp_message.content

    return "Max iterations reached."



def read_remote_csv(url: str) -> str:
    try:
        df = pl.read_csv(url)
        return str(df.head(5))
    except Exception as e:
        return f"Failed to read CSV: {str(e)}"


def read_remote_parquet(url: str) -> str:
    try:
        df = pl.read_parquet(url)
        return str(df.head(5))
    except Exception as e:
        return f"Failed to read Parquet: {str(e)}"


def get_tool_definitions() -> tuple[list[dict], dict[str, Callable]]:
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "read_remote_csv",
                "description": "Reads a CSV file from a URL and returns the content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"}
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_remote_parquet",
                "description": "Reads a Parquet file from a URL and returns the content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"}
                    },
                    "required": ["url"],
                },
            },
        }
    ]

    tool_mapping = {
        "read_remote_csv": read_remote_csv,
        "read_remote_parquet": read_remote_parquet,
    }

    return tool_definitions, tool_mapping


if __name__ == "__main__":
    csv_url = "https://raw.githubusercontent.com/plotly/datasets/master/iris-data.csv"
    print(f"Request: What are the column names in {csv_url}?\nResponse:\n{make_llm_request(f'What are the column names in {csv_url}?')}\n")

    parquet_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    print(
        f"Request: Read the first few rows of {parquet_url} and tell me the VendorID of the first row.\nResponse:\n{make_llm_request(f'Read the first few rows of {parquet_url} and tell me the VendorID of the first row.')}")