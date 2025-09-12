from transformers import AutoTokenizer
import requests
import time
import threading
import numpy as np
import random
import copy
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/models/Qwen2-7B-Instruct")
    parser.add_argument("--model_name", type=str, default="Qwen2-7B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=15)
    parser.add_argument("--is_tensorrt_llm", action="store_true", default=False)
    parser.add_argument("--input_tokens", type=int, default=605)
    parser.add_argument("--session_num", type=int, default=20000)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--mode", type=str, default="chat")
    return parser.parse_args()

def diff(predict, target):
    l2_relative_error = torch.norm(predict - target, p=2) / torch.norm(target, p=2)
    rmse = torch.sqrt(torch.mean((predict - target) ** 2))
    rms = torch.sqrt(torch.mean((target) ** 2))
    similarity = torch.nn.functional.cosine_similarity(predict.reshape(-1), target.reshape(-1), dim=0)
    print(f"rmse: {rmse.item()}, rms: {rms.item()}, l2_relative_error: {l2_relative_error.item()}, similarity: {similarity}")
    return l2_relative_error, rmse, rms, similarity

class Throughput:
    def __init__(self, model_path, model_name, max_tokens, is_tensorrt_llm, input_tokens, session_num, port, mode):
        self.max_tokens = max_tokens
        self.is_tensorrt_llm = is_tensorrt_llm
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.words = []
        for _ in range(20992):
            self.words.append(self.generate_unique_chinese())
        self.word_idx = 0
        self.session = requests.Session()
        self.headers = {
            "Content-Type": "application/json"
        }
        self.mode = mode
        if is_tensorrt_llm:
            self.url = f"http://localhost:{port}/v2/models/tensorrt_llm_bls/generate"
            self.data = {
                "text_input": "如何复现deepseek r1中的知识蒸馏" * (input_tokens // 11),
                "max_tokens": max_tokens
            }
        else:
            if self.mode == "chat":
                self.url = f"http://localhost:{port}/v1/chat/completions"
                self.data = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "如何复现deepseek r1中的知识蒸馏" * (input_tokens // 11)}],
                    "max_tokens": max_tokens,
                    # "chat_template_kwargs": {"enable_thinking": False}
                }
            elif self.mode == "completion":
                self.url = f"http://localhost:{port}/v1/completions"
                self.data = {
                    "model": model_name,
                    "prompt": "如何复现deepseek r1中的知识蒸馏" * (input_tokens // 11),
                    "max_tokens": max_tokens,
                }
            elif self.mode == "openai":
                from openai import OpenAI
                self.client = OpenAI(
                    api_key="EMPTY",
                    base_url="http://localhost:8000/v1",
                )
                import io
                import base64
                prompt_embeds = torch.load("/root/repo/newest/vllm/prompt_embeds.pt")
                buffer = io.BytesIO()
                torch.save(prompt_embeds, buffer)
                buffer.seek(0)
                binary_data = buffer.read()
                encoded_embeds = base64.b64encode(binary_data).decode("utf-8")
                self.client = OpenAI(
                    api_key="EMPTY",
                    base_url="http://localhost:8000/v1",
                )
                self.data = {
                    "model": model_name,
                    "prompt": "",
                    "max_tokens": max_tokens,
                    "prompt_embeds": encoded_embeds,
                    "input_tokens": prompt_embeds.shape[0],
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
            # print(f"request time: {(t2 - t1) * 1000:.2f} ms")
            # print(response_json["outputs"][0]["shape"])
            # prompt = self.data["text_input"]
            # input_tokens = len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt)))
            # out_text = response_json.get("text_output", "")
            # output_tokens = len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(out_text)))
            input_tokens = 0
            output_tokens = 0
            with self.lock:
                results.append((input_tokens, output_tokens, (t2 - t1) * 1000))
            return response_json
        else:
            with self.lock:
                data_tmp = copy.deepcopy(self.data)
                if self.mode == "chat":
                    data_tmp["messages"][0]["content"] = self.words[self.word_idx] + self.data["messages"][0]["content"]
                elif self.mode == "completion":
                    data_tmp["prompt"] = self.words[self.word_idx] + self.data["prompt"]
                self.word_idx += 1
                if self.word_idx % (self.session_num // 5) == 0:
                    print(f"word_idx: {self.word_idx}")
                self.word_idx %= self.session_num
            t1 = time.time()
            if self.mode != "openai":
                response = self.session.post(self.url, headers=self.headers, json=data_tmp)
            else:
                response = self.client.completions.create(
                    model=self.data["model"],
                    prompt=self.data["prompt"],
                    max_tokens=self.data["max_tokens"],
                    extra_body={"prompt_embeds": self.data["prompt_embeds"]},
                )
            t2 = time.time()
            response_json = response.json()
            if self.mode == "chat":
                prompt = data_tmp["messages"][0]["content"]
            elif self.mode == "completion":
                prompt = data_tmp["prompt"]
            if self.mode == "completion":
                input_tokens = len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt)))
            elif self.mode == "chat":
                input_tokens = response_json.get("usage", {}).get("prompt_tokens", 0)
            else:
                input_tokens = self.data["input_tokens"]
            if self.mode == "chat":
                out_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif self.mode == "completion":
                out_text = response_json.get("choices", [{}])[0].get("text", "")
            elif self.mode == "openai":
                out_text = response.choices[0].text
            # reasoning_content = response_json.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "")
            output_tokens = len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(out_text)))
            assert output_tokens == response_json.get("usage", {}).get("completion_tokens", 0)
            # print(f"prompt: {prompt}, out_text: {out_text}, reasoning_content: {reasoning_content}")
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
        median_rt = np.median([result[2] for result in results])
        mean_rt = np.mean([result[2] for result in results])
        return input_tokens, output_tokens, median_rt, mean_rt

    def test_diff_concurrency(self, concurrency_levels, duration):
        # warmup
        self.test_concurrency(1, 1)

        for level in concurrency_levels:
            t1 = time.time()
            input_tokens, output_tokens, median_rt, mean_rt = self.test_concurrency(level, duration)
            t2 = time.time()
            print(f"concurrency: {level}, cost time: {t2 - t1:.2f} s, "
                  f"median_rt: {median_rt:.2f} ms, "
                  f"mean_rt: {mean_rt:.2f} ms, "
                  f"theoretical QPS: {level * (1000 / median_rt):.2f}, "
                  f"avg QPS: {sum(output_tokens) / self.max_tokens / (t2 - t1):.2f}, "
                  f"avg input tokens: {int(np.mean(input_tokens))}, "
                  f"avg output tokens: {int(np.mean(output_tokens))}")

    def test_request(self):
        results = []
        self.send_request(results)
        return results

    def test_embedding(self):
        def request_triton():
            self.url = "http://localhost:8000/v2/models/bls_sync/infer"
            self.data = {
                "inputs": [
                {
                    "name": "id",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["123"]
                },
                {
                    "name": "image_url_list",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["https://china-img.soulapp.cn/image/2024-03-26/29ad3ce7-5316-4789-b497-0c6b185bd23e-1711447302524.png"]
                },
                {
                    "name": "is_base64",
                    "shape": [1],
                    "datatype": "BOOL",
                    "data": [False]
                }
                ],
                "outputs": [
                {"name": "embedding"}
                ]
            }
            results = []
            response_json = self.send_request(results)
            embedding = torch.tensor(response_json["outputs"][0]["data"]).reshape(response_json["outputs"][0]["shape"])
            return embedding

        def request_origin():
            throughput.url = "http://localhost:8080/image_cls_emb_infer"
            throughput.data = {
                "id": "123",
                "image_url_list": ["https://china-img.soulapp.cn/image/2024-03-26/29ad3ce7-5316-4789-b497-0c6b185bd23e-1711447302524.png"],
                "is_base64": False
            }
            results = []
            t1 = time.time()
            response_json = self.send_request(results)
            t2 = time.time()
            print(f"request time: {(t2 - t1) * 1000:.2f} ms")
            r = response_json["image_emb"]
            embedding = torch.tensor(list(r.values()))
            return embedding

        embedding_triton = request_triton()
        embedding_origin = request_origin()
        print(embedding_triton.shape, embedding_triton.dtype, embedding_triton.device)
        print(embedding_origin.shape, embedding_origin.dtype, embedding_origin.device)
        diff(embedding_triton, embedding_origin)

    def test_audio(self):
        self.is_tensorrt_llm = True
        throughput.url = "http://localhost:8000/audio_emb_infer"
        throughput.data = {
            "id": "123",
            "audio_url_list": ["https://soul-app.oss-cn-hangzhou.aliyuncs.com/audio/2024-04-11/2425ec93-f20e-471b-9892-76ff300cd84b.m4a"],
            "is_base64": False
        }
        results = []
        t1 = time.time()
        response_json = self.send_request(results)
        t2 = time.time()
        print(f"request time: {(t2 - t1) * 1000:.2f} ms")
        r = response_json["audio_emb"]
        embedding = torch.tensor(list(r.values()))
        print(embedding.shape, embedding.dtype, embedding.device)
        return embedding

    def test_audio_diff_concurrency(self):
        self.is_tensorrt_llm = True
        self.url = "http://localhost:8000/audio_emb_infer"
        self.data = {
            "id": "123",
            "audio_url_list": ["https://soul-app.oss-cn-hangzhou.aliyuncs.com/audio/2024-04-11/2425ec93-f20e-471b-9892-76ff300cd84b.m4a"],
            "is_base64": False
        }
        concurrency_levels = [16]
        duration = 20
        self.test_diff_concurrency(concurrency_levels, duration)

    def test_image_diff_concurrency(self):
        self.is_tensorrt_llm = True
        self.url = "http://localhost:8080/image_cls_emb_infer"
        self.data = {
            "id": "123",
            "image_url_list": ["https://china-img.soulapp.cn/image/2024-03-26/29ad3ce7-5316-4789-b497-0c6b185bd23e-1711447302524.png"],
            "is_base64": False
        }
        concurrency_levels = [1]
        duration = 20
        self.test_diff_concurrency(concurrency_levels, duration)

    def test_pai_image_diff_concurrency(self):
        self.is_tensorrt_llm = True
        self.url = "http://1063930375378323.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/szfs_image_cls_embedding_ai_clone/image_cls_emb_infer"
        self.data = {
            "id": "123",
            "image_url_list": ["https://china-img.soulapp.cn/image/2024-03-26/29ad3ce7-5316-4789-b497-0c6b185bd23e-1711447302524.png"],
            "is_base64": False
        }
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "ZDExMjc1OTVhNmQwMWM2MTU4NTM3NTUzZTY1MjA2ZTc4ZmQzNWJjOA=="
        }
        concurrency_levels = [1]
        duration = 5
        self.test_diff_concurrency(concurrency_levels, duration)


if __name__ == "__main__":
    args = parse_args()
    throughput = Throughput(args.model_path, args.model_name, args.max_tokens, args.is_tensorrt_llm, args.input_tokens, args.session_num, args.port, args.mode)
    # throughput.test_audio()
    # throughput.test_embedding()
    # throughput.test_audio_diff_concurrency()
    # throughput.test_image_diff_concurrency()
    # throughput.test_pai_image_diff_concurrency()
    # throughput.test_request()
    throughput.test_diff_concurrency([1, 2, 4, 8, 16, 32, 64], 5)

# python test_throughput.py --model_path /root/models/Qwen3-8B --model_name Qwen3-8B --max_tokens 15 --input_tokens 605 --port 8010