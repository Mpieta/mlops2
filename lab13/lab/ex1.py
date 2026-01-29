import time
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

prompts = [
    "What are the benefits of LLMOps?",
    "Explain PagedAttention in three sentences.",
    "How does continuous batching work?",
    "Write a short poem about AI inference.",
    "What is the difference between FP16 and INT4?",
    "Summarize the importance of model quantization.",
    "Explain the KV cache in simple terms.",
    "How does vLLM handle high throughput?",
    "What is bitsandbytes?",
    "Give me a 10-step guide to deploying an LLM."
]

def benchmark():
    start_time = time.time()
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/10...")
        client.chat.completions.create(
            model="Qwen/Qwen3-1.7B",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    total_time = benchmark()
    print(f"\nTotal inference time for 10 prompts: {total_time:.2f} seconds")