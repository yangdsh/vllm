# parallel --sshlogin node1 node2 ./setup.sh
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
# git clone https://github.com/vllm-project/vllm.git
cd ..
python python_only_dev.py

vllm serve meta-llama/Llama-3.1-70B --tensor-parallel-size 4 --pipeline-parallel-size 2