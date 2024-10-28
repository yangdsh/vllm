import os
import time
import signal
import subprocess
import re

# Path to vllm.log
LOG_FILE = "vllm.log"
# Path to the vLLM server command and client
VLLM_SERVER_CMD_TEMPLATE = "vllm serve meta-llama/Llama-3.1-70B --tensor-parallel-size {} --pipeline-parallel-size {} --gpu_memory_utilization 0.99 --max_model_len 8192 > {}"
CLIENT_CMD_TEMPLATE = "python vllm/examples/api_client.py --prompt-length {} --max-tokens {} --host localhost --port 8000 --num-threads {} --n 1"
SERVER_READY_PATTERN = r"Route: /v1/embeddings, Methods: POST"

# Regex pattern to capture the throughput from log lines
THROUGHPUT_PATTERN = r"Avg prompt throughput:\s*([\d\.]+)\s*tokens/s,\s*Avg generation throughput:\s*([\d\.]+)\s*tokens/s"

# Tensor-parallel-size and pipeline-parallel-size combinations
combinations = [(t, p) for t in [8] for p in [1, 2, 4, 8] if t * p >= 16 and t * p <= 16]
prompt_lengths = [4096]
generate_length = 256
num_threads_ = [32]


def run_server(tensor_parallel_size, pipeline_parallel_size, log_file_name):
    """Start the vLLM server with the given parallel sizes."""
    print(f"Starting vLLM server with tensor-parallel-size={tensor_parallel_size}, pipeline-parallel-size={pipeline_parallel_size}")
    server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(tensor_parallel_size, pipeline_parallel_size, log_file_name)
    server_process = subprocess.Popen(server_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return server_process


def wait_for_server_ready(log_file_name):
    """Wait until the server log shows that it's ready by detecting the specific log entry."""
    print("Waiting for the server to be ready...")
    i = 0
    while i < 150:
        if os.path.exists(log_file_name):
            with open(log_file_name, "r") as log_file:
                log_data = log_file.read()
                if re.search(SERVER_READY_PATTERN, log_data):
                    print("Server is ready.")
                    return 1
        time.sleep(1)  # Wait a second before checking again
        i += 1
    return 0


def run_client(prompt_length: int, max_tokens: int, num_threads: int):
    """Run the client with the given prompt and max tokens."""
    print(f"Starting client with prompt of length {prompt_length} and max tokens {max_tokens}...")
    client_cmd = CLIENT_CMD_TEMPLATE.format(prompt_length, max_tokens, num_threads)
    client_process = subprocess.Popen(client_cmd, shell=True)
    client_process.wait()


def extract_floats_from_log_line(line):
    """Extract float values from a log line containing throughput data."""
    try:
        # Split the line into sections by searching for key throughput phrases
        prompt_str = "Avg prompt throughput:"
        generation_str = "Avg generation throughput:"
        
        # Find the position of throughput phrases in the line
        prompt_start = line.find(prompt_str) + len(prompt_str)
        generation_start = line.find(generation_str) + len(generation_str)
        
        # Extract the float values right after the phrases
        avg_prompt_throughput = float(line[prompt_start:].split()[0])  # First float after "Avg prompt throughput:"
        avg_generation_throughput = float(line[generation_start:].split()[0])  # First float after "Avg generation throughput:"

        return avg_prompt_throughput, avg_generation_throughput
    except (IndexError, ValueError) as e:
        # If something goes wrong, return None
        return None, None


def extract_average_throughput_from_log(log_file_name):
    """Extract the average Avg prompt and Avg generation throughput from the log file."""
    print(f"Reading log file: {log_file_name}")
    total_prompt_throughput = 0
    total_generation_throughput = 0
    count = 0  # To keep track of how many throughput entries we find

    with open(log_file_name, "r") as f:
        for line in f:
            if "Avg prompt throughput" in line and "Avg generation throughput" in line:
                # Extract the floats from the line
                prompt_throughput, generation_throughput = extract_floats_from_log_line(line)
                
                if prompt_throughput and generation_throughput:
                    # Accumulate the throughput values
                    total_prompt_throughput += prompt_throughput
                    total_generation_throughput += generation_throughput
                    count += 1

    # Compute the average values if there were any valid entries
    if count > 0:
        avg_prompt_throughput = total_prompt_throughput / count
        avg_generation_throughput = total_generation_throughput / count
        return avg_prompt_throughput, avg_generation_throughput
    else:
        print("No valid throughput data found in the log.")
        return None, None

def extract_highest_throughput_from_log(log_file_name):
    """Extract the highest Avg prompt and Avg generation throughput from the log file."""
    print(f"Reading log file: {log_file_name}")
    max_prompt_throughput = 0
    max_generation_throughput = 0
    with open(log_file_name, "r") as f:
        for line in f:
            if "Avg prompt throughput" in line and "Avg generation throughput" in line:
                # Extract the floats from the line
                prompt_throughput, generation_throughput = extract_floats_from_log_line(line)
                
                if prompt_throughput and generation_throughput:
                    # Update the max values if current ones are higher
                    if prompt_throughput > max_prompt_throughput:
                        max_prompt_throughput = prompt_throughput
                    if generation_throughput > max_generation_throughput:
                        max_generation_throughput = generation_throughput

    # Return the highest values found
    if max_prompt_throughput > 0 or max_generation_throughput > 0:
        return max_prompt_throughput, max_generation_throughput
    else:
        print("No valid throughput data found in the log.")
        return None, None


# Regex pattern to capture the PID from log lines
PID_PATTERN = r"Started engine process with PID (\d+)"

def get_pid_on_port(port):
    """Find the PID of the process that is listening on the given port."""
    try:
        result = subprocess.check_output(["lsof", "-t", f"-i:{port}"])
        pid = int(result.strip())
        print(f"Found PID {pid} on port {port}")
        return pid
    except subprocess.CalledProcessError:
        print(f"No process found on port {port}")
        return None


def kill_server(port=8000):
    """Kill the process running on the given port."""
    pid = get_pid_on_port(port)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)  # Send SIGTERM to terminate the process gracefully
            print(f"Killed process with PID {pid} on port {port}")
            time.sleep(5)
        except ProcessLookupError:
            print(f"Process with PID {pid} not found.")
    else:
        print(f"No process found to kill on port {port}.")


def clean_log_file(log_file_name):
    """Clean up the log file before starting a new run."""
    if os.path.exists(log_file_name):
        os.remove(log_file_name)


def save_results_to_file(results, file_path):
    """Save the benchmark results to a file."""
    with open(file_path, 'a') as f:
        for result in results:
            f.write(",".join(map(str, result)) + "\n")


if __name__ == "__main__":
    results = []
    kill_server()

    # Iterate over each combination of tensor-parallel-size and pipeline-parallel-size
    for tensor_parallel_size, pipeline_parallel_size in combinations:
        # Test different prompt lengths
        for prompt_length in prompt_lengths:
            for num_threads in num_threads_:
                log_file_name = LOG_FILE + str(tensor_parallel_size) + '.' + str(pipeline_parallel_size) + '.' + str(prompt_length)
                clean_log_file(log_file_name)  # Clean the log before each run

                # Start the server with the current configuration
                run_server(tensor_parallel_size, pipeline_parallel_size, log_file_name)
                
                # Wait for the server to be ready before running the client
                wait_for_server_ready(log_file_name)

                for repeat in range(5):
                    # Run the client with the generated prompt and max_tokens = prompt_length
                    run_client(prompt_length, generate_length, num_threads)

                    # Extract throughput numbers from the log
                    prompt_throughput, generation_throughput = extract_average_throughput_from_log(log_file_name)
                    if prompt_throughput is not None and generation_throughput is not None:
                        results.append((tensor_parallel_size, pipeline_parallel_size, prompt_length, num_threads, prompt_throughput, generation_throughput))
                        save_results_to_file(results[-1:], "benchmark_results.csv")
                        print(f"Threads:{num_threads} Results for tensor-parallel-size={tensor_parallel_size}, pipeline-parallel-size={pipeline_parallel_size}, prompt_length={prompt_length}: "
                            f"Prompt throughput = {prompt_throughput} tokens/s, Generation throughput = {generation_throughput} tokens/s")
                    else:
                        print(f"Failed to extract throughput for tensor-parallel-size={tensor_parallel_size}, pipeline-parallel-size={pipeline_parallel_size}, prompt_length={prompt_length}")

                # Kill the server after the client is done
                kill_server()

    # Save results to file
    print(f"\nBenchmark results saved to {"benchmark_results.csv"}")