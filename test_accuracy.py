from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import os
from transformers.generation.utils import GenerationConfig

os.environ["VLLM_USE_V1"] = "0"


class Accuracy:

    def __init__(self, model_path, device, init_hf=True, is_tensorrt_llm=False, dtype_hf="auto"):
        self.model_path = model_path
        self.device = device
        if init_hf:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype_hf,
                                                              device_map=None).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.is_tensorrt_llm = is_tensorrt_llm

    def hf_generate(self, prompt, max_tokens=20):
        generation_config = GenerationConfig(
            top_k=1, temperature=1, max_length=4096,
            max_new_tokens=max_tokens, repetition_penalty=1.0,
            early_stopping=False, do_sample=True, num_beams=1, top_p=1, pad_token_id=0, eos_token_id=2
        )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, generation_config=generation_config)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        out_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return out_text
    
    def request_trtllm(self, prompt, max_tokens=20):
        url = "http://localhost:8000/v2/models/tensorrt_llm_bls/generate"
        data = {
            "text_input": prompt,
            "max_tokens": max_tokens,
            "top_k": 1
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        out_text = response_json["text_output"]
        return out_text
    
    def request_vllm_completions(self, prompt, max_tokens=20):
        model_name = "Qwen2-7B-Instruct"
        data = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "top_k": 1,
            "seed": 41,
            "top_p": 1,
            "temperature": 1.0,
            "repetition_penalty": 1.0
        }
        url = "http://localhost:8010/v1/completions"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        out_text = response_json["choices"][0]["text"]
        return out_text
    
    def request_vllm_chat(self, prompt, max_tokens=20):
        model_name = "Qwen2-7B-Instruct"
        data = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "max_tokens": max_tokens,
            "top_k": 1,
            "seed": 41,
            "top_p": 1,
            "temperature": 1.0,
            "repetition_penalty": 1.0
        }
        url = "http://localhost:8010/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        out_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        return out_text


    def send_request_chat(self, prompt, max_tokens=20):
        if self.is_tensorrt_llm:
            prompt = self.apply_chat_template(prompt)
            out_text = self.request_trtllm(prompt, max_tokens)
        else:
            out_text = self.request_vllm_chat(prompt, max_tokens)
        return out_text


    def send_request_completions(self, prompt, max_tokens=20):
        if self.is_tensorrt_llm:
            out_text = self.request_trtllm(prompt, max_tokens)
        else:
            out_text = self.request_vllm_completions(prompt, max_tokens)
        return out_text

    def test_chat(self, prompt, num_samples=10):
        for _ in range(num_samples):
            input_text = self.apply_chat_template(prompt)
            out_text_hf = self.hf_generate(input_text)
            print(out_text_hf)
        print("--------------------------------------------------------------------")
        for _ in range(num_samples):
            out_text_vllm = self.send_request_chat(prompt)
            print(out_text_vllm)
        print("--------------------------------------------------------------------")

    def apply_chat_template(self, prompt):
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text

    def test_completions(self, prompt):
        out_text_hf = self.hf_generate(prompt)
        print("  hf:", out_text_hf)
        out_text_vllm = self.send_request_completions(prompt)
        print("vllm:",out_text_vllm)

    def split_text(self, text):
        input_ids = self.tokenizer(text)["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        words = []
        convert_tokens = []
        for token in tokens:
            convert_tokens.append(token)
            text = self.tokenizer.convert_tokens_to_string(convert_tokens)
            if "�" in text:
                continue
            else:
                words.append(text)
                convert_tokens = []
        # print(len(input_ids), input_ids)
        # print(len(words), words)
        return words

    def accuracy(self, prompt, enable_chat_template=True):
        if enable_chat_template:
            prompt = self.apply_chat_template(prompt)
        out_text_hf = self.hf_generate(prompt)
        words_hf = self.split_text(out_text_hf)
        hit = 0
        miss = 0
        words_vllm = []
        for word in words_hf:
            out_word_vllm = self.send_request_completions(prompt, 1)
            words_vllm.append(out_word_vllm)
            if out_word_vllm == word:
                hit += 1
            else:
                miss += 1
            prompt = prompt + word
        print(f"Hit: {hit}, Miss: {miss}")
        print("words_hf:", words_hf)
        print("words_vllm:", words_vllm)
        return hit, miss

    def accuracy_all(self, texts, enable_chat_template=True):
        hits = []
        misses = []
        for prompt in texts:
            hit, miss = self.accuracy(prompt, enable_chat_template)
            hits.append(hit)
            misses.append(miss)
        print("Total Hits:", sum(hits))
        print("Total Misses:", sum(misses))
        print(f"Accuracy: {sum(hits) / (sum(hits) + sum(misses)):.4f}")

    def get_texts(self, json_path, text_key='context'):
        import json
        texts = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if text_key in obj:
                        texts.append(obj[text_key])
                except Exception:
                    continue
        return texts


if __name__ == "__main__":
    model_path = "/root/models/Qwen2-7B-Instruct"
    device = "cuda:5"
    init_hf = True
    is_tensorrt_llm = False
    dtype_hf = "float16"
    accuracy_tester = Accuracy(model_path, device, init_hf, is_tensorrt_llm, dtype_hf)
    prompt = "如何复现deepseek r1中的知识蒸馏"

    texts = accuracy_tester.get_texts("/root/repo/soul-llm-evaluate/examples/westworld/npc_test_200.json", "context")
    # texts = accuracy_tester.get_texts("/root/repo/soul-llm-evaluate/examples/qwen2_200.json", "message")
    # accuracy_tester.test_completions(prompt)
    # accuracy_tester.test_chat(prompt, 1)
    accuracy_tester.accuracy_all(texts[:20], enable_chat_template=True)


# vllm serve /root/models/Qwen2-7B-Instruct --served-model-name Qwen2-7B-Instruct --no-enable-prefix-caching --port 8010 --dtype float16
# vllm serve /root/models/Qwen2-7B-Instruct-W8A8-Dynamic-Per-Token --served-model-name Qwen2-7B-Instruct --no-enable-prefix-caching --port 8010
# vllm serve /root/repo/TensorRT-Model-Optimizer/examples/llm_ptq/Qwen2-7B-Instruct-FP8 --served-model-name Qwen2-7B-Instruct --no-enable-prefix-caching --port 8010
