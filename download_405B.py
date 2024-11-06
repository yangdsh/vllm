from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8")
tokenizer = AutoTokenizer.from_pretrained("neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8")