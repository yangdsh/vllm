import argparse
import json
from typing import Iterable, List
import requests
import threading
from datasets import load_dataset

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False,
                      max_tokens: int = 4096) -> requests.Response:
    headers = {"Content-Type": "application/json"}
    pload = {
        "model": "meta-llama/Llama-3.1-70B",
        "prompt": prompt,
        "n": n,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    print(data)
    return data['choices']


def thread_worker(prompt: str, api_url: str, n: int, stream: bool, max_tokens: int) -> None:
    """Worker function for each thread to send an HTTP request."""
    print(f"Sending request with prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, api_url, n, stream, max_tokens)
    
    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output = get_response(response)
        for i, line in enumerate(output):
            print(f"Beam candidate {i}: {line!r}", flush=True)

# Load the dataset
dataset = load_dataset("fka/awesome-chatgpt-prompts")

def generate_prompt(i, length):
    """Generate a prompt from the dataset. If the prompt is not long enough, repeat it."""
    prompt = dataset['train'][i%170]['prompt']  # Get the first prompt (or use random index)
    
    # Check the length of the prompt
    while len(prompt.split()) < length:
        # Repeat the prompt until it reaches the minimum length
        prompt += " " + prompt
    
    return ' '.join(prompt.split()[:length])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)  # beam search sizes
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--prompt-length", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for the generation")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--num-threads", type=int, default=5, help="Number of threads")
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}/v1/completions"
    n = args.n
    stream = args.stream
    prompt = args.prompt
    max_tokens = args.max_tokens
    num_threads = args.num_threads

    # Create and start multiple threads
    threads = []
    for i in range(num_threads):
        # Each thread can have a slightly different prompt, for example, by appending a thread ID
        if args.prompt_length:
            thread_prompt = generate_prompt(i, args.prompt_length)
        else:
            thread_prompt = prompt
        thread = threading.Thread(target=thread_worker, args=(thread_prompt, api_url, n, stream, max_tokens))
        threads.append(thread)
        thread.start()

    # Join all threads to ensure completion
    for thread in threads:
        thread.join()

    print("All requests have been processed.")