from ptflops import get_model_complexity_info
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained("/root/dynamo/Qwen2-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/root/dynamo/Qwen2-7B-Instruct")
model.eval()

# 正确的输入构造函数
def input_constructor(batch_size, seq_length):
    # 创建模拟输入
    text = ["This is a test sentence."] * batch_size
    inputs = tokenizer(text, 
                       padding='max_length', 
                       max_length=seq_length, 
                       return_tensors="pt")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    return inputs['input_ids']

# 计算FLOPs
macs, params = get_model_complexity_info(
    model, 
    input_res=(1, 605),  # 批处理通过构造函数处理，此处仅为占位
    input_constructor=lambda _: input_constructor(batch_size=1, seq_length=605),
    as_strings=False,
    print_per_layer_stat=True
)

print(f"GFLOPs: {2 * macs / 1000 / 1000 / 1000:.2f}")
print(f"Params/M: {params / 1000 / 1000:.2f}")