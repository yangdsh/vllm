# parallel --sshlogin node1 node2 ./setup.sh
# git clone https://github.com/vllm-project/vllm.git

echo "from huggingface_hub import login; login(token='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')" | python
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python python_only_dev.py

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