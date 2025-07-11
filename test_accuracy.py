from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import os

os.environ["VLLM_USE_V1"] = "0"


class Accuracy:

    def __init__(self, model_path, device, init_hf=True):
        self.model_path = model_path
        self.device = device
        if init_hf:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",
                                                              device_map=None).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def hf_generate(self, prompt, max_tokens=20):
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=max_tokens, top_k=1)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def send_request_chat(self, prompt, max_tokens=20):
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
        }
        url = "http://localhost:8010/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        out_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        return out_text

    def send_request_completions(self, prompt, max_tokens=20):
        model_name = "Qwen2-7B-Instruct"
        data = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "top_k": 1,
            "seed": 41,
        }
        url = "http://localhost:8010/v1/completions"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        out_text = response_json["choices"][0]["text"]
        return out_text

    def test_chat(self, prompt, num_samples=10):
        for _ in range(num_samples):
            out_text_hf = self.hf_generate(prompt)
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

    def test_chat2(self, prompt):
        text = self.apply_chat_template(prompt)
        out_text_vllm = self.send_request_completions(text)
        print(out_text_vllm)

    def split_text(self, text):
        input_ids = self.tokenizer(text)["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        words = []
        for token in tokens:
            text = self.tokenizer.convert_tokens_to_string([token])
            words.append(text)
        # print(input_ids)
        # print(words)
        return words

    def accuracy(self, prompt):
        out_text_hf = self.hf_generate(prompt)
        words_hf = self.split_text(out_text_hf)
        hit = 0
        miss = 0
        words_vllm = []
        prompt = self.apply_chat_template(prompt)
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

    def get_texts(self, json_path):
        import json
        texts = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if 'context' in obj:
                        texts.append(obj['context'])
                except Exception:
                    continue
        return texts


if __name__ == "__main__":
    model_path = "/root/models/Qwen2-7B-Instruct"
    device = "cuda:2"
    init_hf = True
    accuracy_tester = Accuracy(model_path, device, init_hf)
    prompt = "如何复现deepseek r1中的知识蒸馏"

    texts = accuracy_tester.get_texts("/root/repo/soul-llm-evaluate/examples/westworld/npc_test_200.json")
    # prompt = texts[0]
    # accuracy_tester.test_chat2(prompt)
    # accuracy_tester.test_chat(prompt, 1)

    hits = []
    misses = []
    for prompt in texts:
        hit, miss = accuracy_tester.accuracy(prompt)
        hits.append(hit)
        misses.append(miss)
    print("Total Hits:", sum(hits))
    print("Total Misses:", sum(misses))
    print(f"Accuracy: {sum(hits) / (sum(hits) + sum(misses)):.2f}")
