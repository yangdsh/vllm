# parallel --sshlogin node1 node2 ./setup.sh
# git clone https://github.com/vllm-project/vllm.git

echo "from huggingface_hub import login; login(token='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')" | python
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python python_only_dev.py
pip install 'accelerate>=0.26.0'

vllm serve meta-llama/Llama-3.1-70B --tensor-parallel-size 4 --pipeline-parallel-size 2


export NCCL_SOCKET_IFNAME="!rdmap1*"
mpirun -x FI_EFA_USE_DEVICE_RDMA=1 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH --hostfile my-hosts -n 16 -N 8 ~/nccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2

python benchmarks/benchmark_serving.py \
    --result-dir results \
    --save-result \
    --result-filename results.json \
    --model meta-llama/Llama-3.1-70B \
    --dataset-name random \
    --num-prompts 128 \
    --random-input-len 1024 \
    --request-rate 50

# python benchmarks/benchmark_serving.py --model neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8 --dataset-name random --num-prompts 32 --random-input-len 1024 --request-rate 1

python benchmarks/benchmark_serving.py --model Qwen/Qwen2.5-32B --dataset-name sharegpt --dataset-path ~/BurstGPT/example/preprocess_data/shareGPT.json --num-prompts 128 --random-input-len 1024 --request-rate 8 --max-concurrency 8 --input-lens 128 1024 2048 --input-ratios 0.6 0.3 0.1 

FI_EFA_USE_DEVICE_RDMA=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH NCCL_DEBUG=INFO vllm serve meta-llama/Llama-3.1-70B --disable_custom_all_reduce --max-model-len 16384 --enable-chunked-prefill --tensor-parallel-size 4 --pipeline-parallel-size 2