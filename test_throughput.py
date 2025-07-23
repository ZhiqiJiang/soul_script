from transformers import AutoTokenizer
import requests
import time
import threading
import numpy as np
import random
import copy

class Throughput:
    def __init__(self, model_path, model_name, max_tokens, is_tensorrt_llm, session_num=20000):
        self.model_path = model_path
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.is_tensorrt_llm = is_tensorrt_llm
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.words = []
        for i in range(20992):
            self.words.append(self.generate_unique_chinese())
        self.word_idx = 0
        self.session = requests.Session()
        self.headers = {
            "Content-Type": "application/json"
        }
        if is_tensorrt_llm:
            self.url = "http://localhost:8000/v2/models/tensorrt_llm_bls/generate"
            self.data = {
                "text_input": "如何复现deepseek r1中的知识蒸馏" * 55,
                "max_tokens": max_tokens
            }
        else:
            self.url = "http://localhost:8010/v1/chat/completions"
            self.data = {
                "model": model_name,
                "messages": [{"role": "user", "content": "如何复现deepseek r1中的知识蒸馏" * 22}],
                "max_tokens": max_tokens
            }

        self.lock = threading.Lock()
        self.session_num = session_num



    def generate_unique_chinese(self):
        if not hasattr(self, "chars"):
            # 初始化汉字列表并打乱顺序
            start = 0x4E00
            end = 0x9FFF
            self.chars = [
                chr(code_point) for code_point in range(start, end + 1)
            ]
            random.shuffle(self.chars)
        
        if self.chars:
            return self.chars.pop()
        else:
            raise StopIteration("所有汉字已用完")

    def send_request(self, results):
        if self.is_tensorrt_llm:
            t1 = time.time()
            response = self.session.post(self.url, headers=self.headers, json=self.data)
            t2 = time.time()
            response_json = response.json()
            prompt = self.data["text_input"]
            input_tokens = len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt)))
            out_text = response_json.get("text_output", "")
            output_tokens = len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(out_text)))
            with self.lock:
                results.append((input_tokens, output_tokens, (t2 - t1) * 1000))
        else:
            with self.lock:
                data_tmp = copy.deepcopy(self.data)
                data_tmp["messages"][0]["content"] = self.words[self.word_idx] + self.data["messages"][0]["content"]
                self.word_idx += 1
                if self.word_idx % (self.session_num // 5) == 0:
                    print(f"word_idx: {self.word_idx}")
                self.word_idx %= self.session_num
            t1 = time.time()
            response = self.session.post(self.url, headers=self.headers, json=data_tmp)
            t2 = time.time()
            response_json = response.json()
            prompt = data_tmp["messages"][0]["content"]
            input_tokens = len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt)))
            input_tokens = response_json.get("usage", {}).get("prompt_tokens", 0)
            out_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            output_tokens = len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(out_text)))
            assert output_tokens == response_json.get("usage", {}).get("completion_tokens", 0)
            # print(f"request time: {(t2 - t1) * 1000:.2f} ms, input_tokens_num: {input_tokens}, "
            #       f"output_tokens_num: {output_tokens}, TPOT: {(t2 - t1) * 1000 / output_tokens:.2f} ms, "
            #       f"服务器处理时间: {response.elapsed.total_seconds() * 1000:.2f} ms")
            with self.lock:
                results.append((input_tokens, output_tokens, (t2 - t1) * 1000))

    def test_concurrency(self, concurrency_level, duration=5):
        threads = []
        results = []
        start_time = time.time()

        def worker():
            while time.time() - start_time < duration:
                self.send_request(results)

        for _ in range(concurrency_level):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        input_tokens = [result[0] for result in results]
        output_tokens = [result[1] for result in results]
        rt = np.mean([result[2] for result in results])
        return input_tokens, output_tokens, rt

    def test_diff_concurrency(self):
        concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        duration = 5
        for level in concurrency_levels:
            t1 = time.time()
            input_tokens, output_tokens, rt = self.test_concurrency(level, duration)
            t2 = time.time()
            print(f"concurrency: {level}, cost time: {t2 - t1:.2f} s, "
                  f"RT: {rt:.2f} ms, "
                  f"QPS: {sum(output_tokens) / max_tokens / (t2 - t1):.2f}, "
                  f"avg input tokens: {int(np.mean(input_tokens))}, "
                  f"avg output tokens: {int(np.mean(output_tokens))}")

    def test_request(self):
        results = []
        self.send_request(results)
        return results

if __name__ == "__main__":
    model_path = "/root/models/Qwen2-7B-Instruct"
    model_name = "Qwen2-7B-Instruct"
    max_tokens = 6
    is_tensorrt_llm = False
    throughput = Throughput(model_path, model_name, max_tokens, is_tensorrt_llm)
    throughput.test_diff_concurrency()