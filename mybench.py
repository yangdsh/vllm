import os
import time
import subprocess
import re
import math
import psutil

import subprocess
import socket

# Constants for server log and commands
original_pp_partition = True
mix_prompts = False
# MODEL = 'neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8'
# MODEL = 'meta-llama/Llama-3.1-70B'
MODEL = 'Qwen/Qwen2.5-32B'
# MODEL = 'Qwen/Qwen2-72B-Instruct'
DIR = f"results/{MODEL.split('/')[-1]}"
DIR = DIR + "-mix-prompts" if mix_prompts else DIR
if not os.path.exists(DIR):
    os.makedirs(DIR)
LOG_FILE = f"{DIR}/vllm.log"
NUM_LAYERS = 126
DATASET = 'csv' # 'random'

# Argument combinations
server_combinations = [(t, p) for t in [4] for p in [1] if t * p <= 8 and t * p >= 4]
'''client_combinations_ = [(prompt_len, rate * max_concurrency, max_concurrency) for prompt_len in [128, 1024, 2048, 4096, 8192] for rate in [1] for max_concurrency in [32, 64]]
client_combinations = []
for random_input_len, request_rate, max_concurrency in client_combinations_:
    if mix_prompts:
        random_input_len = 2
    else:
        request_rate = request_rate * (1024 / random_input_len)
        if request_rate < 1:
            continue
        request_rate = int(request_rate)
        max_concurrency = int(max_concurrency * (1024 / random_input_len))
    comb = (random_input_len, request_rate, max_concurrency)
    if comb not in client_combinations:
        client_combinations.append(comb)'''
client_combinations = [(prompt_len, rate * max_concurrency, max_concurrency, input_len_range) for prompt_len in [1024] for rate in [1] for max_concurrency in [32] for input_len_range in [[1, 512], [512, 2048], [1, 2048]]]



def restart_ray():
    # Stop Ray
    subprocess.run("ray stop --force", shell=True, check=True)
    # Start Ray (adjust the options as needed)
    subprocess.run(f"VLLM_PP_LAYER_PARTITION={os.environ["VLLM_PP_LAYER_PARTITION"]} RAY_DEDUP_LOGS=0 ray start --head", shell=True, check=True)

def restart_ray_remote(hostname):
    # Get the current node IP (head node IP)
    def get_current_ip():
        # Get the current node's IP address
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    head_node_ip = get_current_ip()

    # Construct the SSH commands to run on the remote node
    stop_command = f"ssh {hostname} 'ray stop --force'"
    start_command = f"ssh {hostname} 'VLLM_PP_LAYER_PARTITION={os.environ["VLLM_PP_LAYER_PARTITION"]} RAY_DEDUP_LOGS=0 ray start --address=\"{head_node_ip}:6379\"'"

    # Execute the commands
    try:
        subprocess.run(stop_command, shell=True, check=True)
        subprocess.run(start_command, shell=True, check=True)
        print(f"Ray successfully restarted on {hostname}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to restart Ray on {hostname}: {e}")

def set_pp_layers(pp):
    if pp <= 2 or original_pp_partition:
        os.environ["VLLM_PP_LAYER_PARTITION"] = ""
        return
    else:
        layer_per_stage = math.ceil(NUM_LAYERS / pp)
        first_last = NUM_LAYERS - (pp - 2) * layer_per_stage
        first = first_last // 2
        last = first_last - first
        layer_partition = [first]
        layer_partition += [layer_per_stage for i in range(pp-2)]
        layer_partition.append(last)
        os.environ["VLLM_PP_LAYER_PARTITION"] = ','.join(str(i) for i in layer_partition)
        os.environ["VLLM_PP_LAYER_PARTITION"] = '31,31,33,31'
    print("layer partition: ", os.environ["VLLM_PP_LAYER_PARTITION"])
    restart_ray()
    restart_ray_remote("node2")


VLLM_SERVER_CMD_TEMPLATE = (#"NCCL_DEBUG=INFO FI_EFA_USE_DEVICE_RDMA=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH "
                            f"vllm serve {MODEL} --disable_custom_all_reduce "
                            "--max-model-len 16384 "
                            "--gpu_memory_utilization 0.95 "
                            "--enable-chunked-prefill "
                            "--tensor-parallel-size {} --pipeline-parallel-size {} "
                            # "--distributed-executor-backend ray "
                            )
MIX = "--input-lens 128 2048 --input-ratios 0.89 0.11 " if mix_prompts else " "
CLIENT_CMD_TEMPLATE = (f"python benchmarks/benchmark_serving.py --result-dir {DIR} --save-result --model {MODEL} --dataset-name {DATASET} --csv-file-path ~/BurstGPT/data/BurstGPT_1.csv --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --sharegpt-output-len 64 --random-output-len 1 {MIX}"
"--result-filename {} --num-prompts {} --random-input-len {} --input-lens {} {} --request-rate {} --max-concurrency {} --host 172.31.85.15 --prompt-len-threshold {}")
SERVER_READY_PATTERN = r"Route: /v1/embeddings, Methods: POST"
CUDA_OOM_PATTERN = r"CUDA out of memory"
ERROR_PATTERN = r"error"
RAISE_PATTERN = r"raise"
skip_starting_server = False



def run_server(tensor_parallel_size, pipeline_parallel_size, hostname):
    """Start the server with specific tensor and pipeline parallel sizes."""
    log_file_name = f"{LOG_FILE}_{tensor_parallel_size}_{pipeline_parallel_size}_{hostname}.log"
    print(f"Starting server with tensor-parallel-size={tensor_parallel_size}, pipeline-parallel-size={pipeline_parallel_size}")
    server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(tensor_parallel_size, pipeline_parallel_size)
    print(server_cmd)
    print(log_file_name)
    with open(log_file_name, "w") as log_file:
        server_process = server_process = subprocess.Popen(f"ssh {hostname} '{server_cmd}'", shell=True, stdout=log_file, stderr=log_file)
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

def run_client(random_input_len, input_len_range, request_rate, max_concurrency, tensor_parallel_size, pipeline_parallel_size):
    """Run the client with specified input length and request rate, saving to file."""
    result_filename = f"results_{tensor_parallel_size}_{pipeline_parallel_size}_{input_len_range[0]}~{input_len_range[1]}_{request_rate}_{max_concurrency}{"_mypp" if not original_pp_partition else ""}.json"
    # run client
    num_prompts = 512
    prompt_len_threshold = 512
    client_cmd = CLIENT_CMD_TEMPLATE.format(result_filename, num_prompts, random_input_len, input_len_range[0], input_len_range[1], request_rate, max_concurrency, prompt_len_threshold)
    print(client_cmd)
    print(f"Running client with input length {input_len_range} and request rate {request_rate}")
    client_process = subprocess.Popen(client_cmd, shell=True)
    client_process.wait()

def is_port_in_use(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False

def kill_server():
    """Kill any running server process."""
    try:
        subprocess.run("pkill -f 'vllm serve'", shell=True, check=True)
        subprocess.run(["ssh", "node2", "pkill -f \\\"vllm serve\\\""], check=True)
        print("Killed any running 'vllm serve' process.")
    except subprocess.CalledProcessError as e:
        print("No 'vllm serve' processes were running:", e)
    time.sleep(10)
    if is_port_in_use(8000):
        kill_server()

def clean_log_file(log_file_name):
    """Remove log file before starting a new server."""
    if os.path.exists(log_file_name):
        os.remove(log_file_name)

if __name__ == "__main__":
    # Loop through server configurations
    for tensor_parallel_size, pipeline_parallel_size in server_combinations:
        set_pp_layers(pipeline_parallel_size)
        if not skip_starting_server:
            kill_server()  # Ensure no previous server instance is running
            log_file_name = f"{LOG_FILE}_{tensor_parallel_size}_{pipeline_parallel_size}.log"
            clean_log_file(log_file_name)

            # Start server and check readiness
            _, log_file_name1 = run_server(tensor_parallel_size, pipeline_parallel_size, "localhost")
            _, log_file_name2 = run_server(2, 4, "node2")
            if not wait_for_server_ready(log_file_name1) or not wait_for_server_ready(log_file_name2):
                print("Skipping to next server configuration due to startup issues.")
                continue

        # Loop through client configurations without restarting the server
        for random_input_len, request_rate, max_concurrency, input_len_range in client_combinations:
            run_client(random_input_len, input_len_range, request_rate, max_concurrency, tensor_parallel_size, pipeline_parallel_size)
        print(f"Completed testing for tensor-parallel-size={tensor_parallel_size}, pipeline-parallel-size={pipeline_parallel_size}\n")