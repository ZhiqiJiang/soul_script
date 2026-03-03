from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import time

model_path = "/root/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=model_path, model_revision="v2.0.4")

bs_range = [1, 2, 4, 8, 16]

def warmup():
    for i in range(10):
        result = inference_pipeline(input=['/root/repo/soul_script/asr_example_zh.wav'])

warmup()

for bs in bs_range:
    i = 0
    if i == 11:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
    t1 = time.time()
    rec_result = inference_pipeline(input=['/root/repo/soul_script/asr_example_zh.wav'] * bs, batch_size=bs)
    t2 = time.time()
    if i == 11:
        pr.disable()
        pr.dump_stats("/root/repo/soul_script/asr.prof")
        import pstats
        ps = pstats.Stats(pr)
        ps.sort_stats('cumulative').print_stats(20)
    print(f"total: {(t2 - t1) * 1000} ms")
    print("-" * 100)
print(rec_result)