import time
import threading
import numpy as np
import argparse
import tritonclient.grpc as grpcclient

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/models/Qwen2-7B-Instruct")
    parser.add_argument("--model_name", type=str, default="Qwen2-7B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=15)
    parser.add_argument("--is_tensorrt_llm", action="store_true", default=False)
    parser.add_argument("--input_tokens", type=int, default=605)
    parser.add_argument("--session_num", type=int, default=20000)
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()

class Throughput:
    def __init__(self, model_path, model_name, max_tokens, is_tensorrt_llm, input_tokens, session_num, port):
        self.lock = threading.Lock()

        transformed_img = np.ones((1, 3, 960, 960)).astype(np.float32)
        self.test_input = grpcclient.InferInput("x", transformed_img.shape, datatype="FP32")
        self.test_input.set_data_from_numpy(transformed_img)
        self.test_output = grpcclient.InferRequestedOutput("save_infer_model/scale_0.tmp_1", class_count=2)


    def send_request(self, results, triton_client):
        t1 = time.time()
        res = triton_client.infer(model_name="general-ppocr-det", inputs=[self.test_input])
        t2 = time.time()
        # test_output_fin = res.as_numpy('save_infer_model/scale_0.tmp_1')
        # print(test_output_fin[:2])
        # print(f"infer: {(t2 - t1) * 1000000:.2f} us")
        with self.lock:
            results.append((t2 - t1) * 1000000)

    def test_concurrency(self, concurrency_level, duration=5):
        threads = []
        results = []
        start_time = time.time()

        def worker(triton_client):
            while time.time() - start_time < duration:
                self.send_request(results, triton_client)
                # time.sleep(0.01)

        triton_clients = []
        for i in range(concurrency_level):
            triton_clients.append(grpcclient.InferenceServerClient(url="localhost:8004"))
            thread = threading.Thread(target=worker, args=(triton_clients[i],))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        rt = np.mean(results)
        return rt

    def test_diff_concurrency(self):
        concurrency_levels = [1, 4]
        duration = 5
        # warmup
        self.test_concurrency(1, 1)

        for level in concurrency_levels:
            t1 = time.time()
            rt = self.test_concurrency(level, duration)
            t2 = time.time()
            print(f"concurrency: {level}, cost time: {t2 - t1:.2f} s, "
                  f"RT: {rt:.2f} us")

    def test_request(self):
        results = []
        triton_client = grpcclient.InferenceServerClient(url="localhost:8004")
        self.send_request(results, triton_client)
        return results

if __name__ == "__main__":
    args = parse_args()
    throughput = Throughput(args.model_path, args.model_name, args.max_tokens, args.is_tensorrt_llm, args.input_tokens, args.session_num, args.port)
    # throughput.test_request()
    throughput.test_diff_concurrency()