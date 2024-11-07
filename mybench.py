import os
import time
import subprocess
import re
import math

import subprocess
import socket

def restart_ray():
    # Stop Ray
    subprocess.run("ray stop --force", shell=True, check=True)
    # Start Ray (adjust the options as needed)
    subprocess.run("ray start --head", shell=True, check=True)

def restart_ray_remote(hostname):
    # Get the current node IP (head node IP)
    def get_current_ip():
        # Get the current node's IP address
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    head_node_ip = get_current_ip()

    # Construct the SSH commands to run on the remote node
    stop_command = f"ssh {hostname} 'ray stop --force'"
    start_command = f"ssh {hostname} 'VLLM_PP_LAYER_PARTITION={os.environ["VLLM_PP_LAYER_PARTITION"]} ray start --address=\"{head_node_ip}:6379\"'"

    # Execute the commands
    try:
        subprocess.run(stop_command, shell=True, check=True)
        subprocess.run(start_command, shell=True, check=True)
        print(f"Ray successfully restarted on {hostname}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to restart Ray on {hostname}: {e}")

def set_pp_layers(pp, num_layers):
    if pp <= 2 or original_pp_partition:
        os.environ["VLLM_PP_LAYER_PARTITION"] = ''
        return
    layer_per_stage = math.ceil(num_layers / pp)
    first_last = num_layers - (pp - 2) * layer_per_stage
    first = first_last // 2
    last = first_last - first
    layer_partition = [first]
    layer_partition += [layer_per_stage for i in range(pp-2)]
    layer_partition.append(last)
    os.environ["VLLM_PP_LAYER_PARTITION"] = ','.join(str(i) for i in layer_partition)
    print("layer partition: ", os.environ["VLLM_PP_LAYER_PARTITION"])
    restart_ray()
    restart_ray_remote("node2")

# Constants for server log and commands
original_pp_partition = False
DIR = "results-405B-request_rates"
NUM_LAYERS = 126
if not os.path.exists(DIR):
    os.makedirs(DIR)
LOG_FILE = f"{DIR}/vllm.log"
MODEL = 'neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8'
# MODEL = 'meta-llama/Llama-3.1-70B'
VLLM_SERVER_CMD_TEMPLATE = ("FI_EFA_USE_DEVICE_RDMA=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH "
                            f"vllm serve {MODEL} --disable_custom_all_reduce "
                            "--max-model-len 4096 "
                            "--gpu_memory_utilization 0.95 "
                            "--enable-chunked-prefill "
                            "--tensor-parallel-size {} --pipeline-parallel-size {} --distributed-executor-backend ray")
CLIENT_CMD_TEMPLATE = (f"python benchmarks/benchmark_serving.py --result-dir {DIR} --save-result --model {MODEL} --dataset-name random "
"--result-filename {} --num-prompts {} --random-input-len {} --request-rate {} --max-concurrency {}")
SERVER_READY_PATTERN = r"Route: /v1/embeddings, Methods: POST"
CUDA_OOM_PATTERN = r"CUDA out of memory"
ERROR_PATTERN = r"error"
RAISE_PATTERN = r"raise"
skip_starting_server = False

# Argument combinations
server_combinations = [(t, p) for t in [1,2,4,8,16] for p in [4] if t * p >= 16 and t * p <= 16]
client_combinations = [(prompt_len, rate * max_concurrency, max_concurrency) for prompt_len in [1024] for rate in [1] for max_concurrency in [10, 20, 30]]  # Adjust prompt lengths and request rates as needed


def run_server(tensor_parallel_size, pipeline_parallel_size):
    """Start the server with specific tensor and pipeline parallel sizes."""
    log_file_name = f"{LOG_FILE}_{tensor_parallel_size}_{pipeline_parallel_size}.log"
    print(f"Starting server with tensor-parallel-size={tensor_parallel_size}, pipeline-parallel-size={pipeline_parallel_size}")
    server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(tensor_parallel_size, pipeline_parallel_size)
    print(server_cmd)
    print(log_file_name)
    with open(log_file_name, "w") as log_file:
        server_process = server_process = subprocess.Popen(server_cmd, shell=True, stdout=log_file, stderr=log_file)
    return server_process, log_file_name

def wait_for_server_ready(log_file_name, timeout=600):
    """Wait for server readiness or detect CUDA OOM within the log."""
    print("Waiting for the server to be ready...")
    for _ in range(timeout):
        if os.path.exists(log_file_name):
            with open(log_file_name, "r") as log_file:
                log_data = log_file.read()
                if re.search(SERVER_READY_PATTERN, log_data):
                    print("Server is ready.")
                    return True
                if re.search(CUDA_OOM_PATTERN, log_data):
                    print("Server OOM.")
                    return False
                if re.search(ERROR_PATTERN, log_data) or re.search(RAISE_PATTERN, log_data):
                    print("Server raised error.")
                    return False
        time.sleep(1)
    print("Server startup timed out.")
    return False

def run_client(random_input_len, request_rate, max_concurrency, tensor_parallel_size, pipeline_parallel_size):
    """Run the client with specified input length and request rate, saving to file."""
    result_filename = f"results_{tensor_parallel_size}_{pipeline_parallel_size}_{random_input_len}_{request_rate}_{max_concurrency}_{"mypp" if original_pp_partition else None}.json"
    # run client
    client_cmd = CLIENT_CMD_TEMPLATE.format(result_filename, max_concurrency * 10, random_input_len, request_rate, max_concurrency)
    print(f"Running client with input length {random_input_len} and request rate {request_rate}")
    client_process = subprocess.Popen(client_cmd, shell=True)
    client_process.wait()

def kill_server():
    """Kill any running server process."""
    try:
        subprocess.run("pkill -f 'vllm serve'", shell=True, check=True)
        print("Killed any running 'vllm serve' process.")
    except subprocess.CalledProcessError as e:
        print("No 'vllm serve' processes were running:", e)

def clean_log_file(log_file_name):
    """Remove log file before starting a new server."""
    if os.path.exists(log_file_name):
        os.remove(log_file_name)

if __name__ == "__main__":
    # Loop through server configurations
    for tensor_parallel_size, pipeline_parallel_size in server_combinations:
        set_pp_layers(pipeline_parallel_size, NUM_LAYERS)
        if not skip_starting_server:
            kill_server()  # Ensure no previous server instance is running
            log_file_name = f"{LOG_FILE}_{tensor_parallel_size}_{pipeline_parallel_size}.log"
            clean_log_file(log_file_name)

            # Start server and check readiness
            server_process, log_file_name = run_server(tensor_parallel_size, pipeline_parallel_size)
            if not wait_for_server_ready(log_file_name):
                print("Skipping to next server configuration due to startup issues.")
                continue

        # Loop through client configurations without restarting the server
        for random_input_len, request_rate, max_concurrency in client_combinations:
            run_client(random_input_len, request_rate, max_concurrency, tensor_parallel_size, pipeline_parallel_size)
        print(f"Completed testing for tensor-parallel-size={tensor_parallel_size}, pipeline-parallel-size={pipeline_parallel_size}\n")