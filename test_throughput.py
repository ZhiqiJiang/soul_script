from transformers import AutoTokenizer
import requests
import time
import threading
import numpy as np
import random
import copy

def generate_unique_chinese():
    if not hasattr(generate_unique_chinese, "chars"):
        # 初始化汉字列表并打乱顺序
        start = 0x4E00
        end = 0x9FFF
        generate_unique_chinese.chars = [
            chr(code_point) for code_point in range(start, end + 1)
        ]
        random.shuffle(generate_unique_chinese.chars)
    
    if generate_unique_chinese.chars:
        return generate_unique_chinese.chars.pop()
    else:
        raise StopIteration("所有汉字已用完")

words = []
for i in range(20992):
    words.append(generate_unique_chinese())
word_idx = 0

session_num = 20000

model_path = "/root/dynamo/Qwen2-7B-Instruct"
model_name = "Qwen2-7B-Instruct"
model_path = "/root/models/Qwen3-30B-A3B-FP8"
model_name = "Qwen3-30B-A3B-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_tokens = 15

# Define the URL and headers
url = "http://localhost:8010/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": model_name,
    "messages": [
        {"role": "user", "content": "如何复现deepseek r1中的知识蒸馏" * 55} # 11 * 55 = 605 tokens
    ],
    "max_tokens": max_tokens
}

# Function to send a request and return the token counts
def send_request(results, lock):
    t1 = time.time()
    with lock:
        global word_idx
        data_tmp = copy.deepcopy(data)
        data_tmp["messages"][0]["content"] = words[word_idx] + data["messages"][0]["content"]
        word_idx += 1
        global session_num
        if word_idx % (session_num // 5) == 0:
            print(f"word_idx: {word_idx}")
        word_idx %= session_num
    response = requests.post(url, headers=headers, json=data_tmp)
    response_json = response.json()
    prompt = data_tmp["messages"][0]["content"]
    out_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
    input_tokens = len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt)))
    input_tokens = response_json.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(out_text)))
    assert output_tokens == response_json.get("usage", {}).get("completion_tokens", 0)
    t2 = time.time()
    # print(f"request time: {(t2 - t1) * 1000:.2f} ms, input_tokens_num: {input_tokens}, "
    #       f"output_tokens_num: {output_tokens}, TPOT: {(t2 - t1) * 1000 / output_tokens:.2f} ms")
    with lock:
        results.append((input_tokens, output_tokens, (t2 - t1) * 1000))

# Function to test different concurrency levels
def test_concurrency(concurrency_level, duration=5):
    threads = []
    results = []
    lock = threading.Lock()
    start_time = time.time()

    def worker():
        while time.time() - start_time < duration:
            send_request(results, lock)

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

def test_diff_concurrency():
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    duration = 5
    for level in concurrency_levels:
        t1 = time.time()
        input_tokens, output_tokens, rt = test_concurrency(level, duration)
        t2 = time.time()
        print(f"concurrency: {level}, cost time: {t2 - t1:.2f} s, "
              f"RT: {rt:.2f} ms, "
              f"QPS: {sum(output_tokens) / max_tokens / (t2 - t1):.2f}, "
              f"avg input tokens: {int(np.mean(input_tokens))}, "
              f"avg output tokens: {int(np.mean(output_tokens))}")

def test_request():
    results = []
    lock = threading.Lock()
    send_request(results, lock)

if __name__ == "__main__":
    test_diff_concurrency()
    # test_request()
