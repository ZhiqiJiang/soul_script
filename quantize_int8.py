# 1. Loading the Model
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "/root/models/Qwen2/Qwen2-7B-Instruct"
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
# ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

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

# Configure the quantization algorithms
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save the compressed model: Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token
SAVE_DIR = "/root/models/Qwen2/Qwen2-7B-Instruct" + "-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
'''
CUDA_VISIBLE_DEVICES=7 \
lm_eval --model vllm \
  --model_args pretrained="/root/models/Qwen2/Qwen2-7B-Instruct",add_bos_token=true \
  --tasks cmmlu \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'

- Qwen2-7B-Instruct
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.784|±  |0.0261|
|     |       |strict-match    |     5|exact_match|↑  |0.700|±  |0.0290|

- Qwen2-7B-Instruct-W8A8-Dynamic-Per-Token
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.748|±  |0.0275|
|     |       |strict-match    |     5|exact_match|↑  |0.656|±  |0.0301|
'''
