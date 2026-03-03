# 1. Loading the Model
import os

# Restrict to GPU 5 and 6 and enable model sharding across them
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "/root/models/Qwen3/Qwen3-30B-A3B"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2. Preparing Calibration Data
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load and preprocess the dataset
ds = load_dataset("/root/models/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# 3. Applying Quantization
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the quantization algorithms
recipe = [
    # SmoothQuantModifier(smoothing_strength=0.5),
    QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head", "re:.*mlp.gate$"]),
]

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    calibrate_moe_context=True,
)

# Save the compressed model: Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token
SAVE_DIR = "/root/models/Qwen3/Qwen3-30B-A3B" + "-W8A8-Dynamic-Per-Token2"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

'''
CUDA_VISIBLE_DEVICES=2 \
lm_eval --model vllm \
  --model_args pretrained="/root/models/Qwen3/Qwen3-30B-A3B-FP8-DYNAMIC",add_bos_token=true,tensor_parallel_size=1 \
  --tasks cmmlu \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'

CUDA_VISIBLE_DEVICES=4,5 \
lm_eval --model vllm \
  --model_args pretrained="/root/models/Qwen3/Qwen3-30B-A3B",add_bos_token=true,tensor_parallel_size=2 \
  --tasks cmmlu \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'


- Qwen3-30B-A3B
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.852|±  |0.0225|
|     |       |strict-match    |     5|exact_match|↑  |0.904|±  |0.0187|

|Groups|Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|------|------:|------|------|--------|---|-----:|---|-----:|
|cmmlu |      1|none  |      |acc     |↑  |0.8346|±  |0.0034|
|      |       |none  |      |acc_norm|↑  |0.8346|±  |0.0034|


- Qwen3-30B-A3B-W8A8-Dynamic-Per-Token
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.848|±  |0.0228|
|     |       |strict-match    |     5|exact_match|↑  |0.908|±  |0.0183|

|Groups|Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|------|------:|------|------|--------|---|-----:|---|-----:|
|cmmlu |      1|none  |      |acc     |↑  |0.8343|±  |0.0034|
|      |       |none  |      |acc_norm|↑  |0.8343|±  |0.0034|


- Qwen3-30B-A3B-FP8-DYNAMIC
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.820|±  |0.0243|
|     |       |strict-match    |     5|exact_match|↑  |0.908|±  |0.0183|

|Groups|Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|------|------:|------|------|--------|---|-----:|---|-----:|
|cmmlu |      1|none  |      |acc     |↑  |0.8344|±  |0.0034|
|      |       |none  |      |acc_norm|↑  |0.8344|±  |0.0034|
'''