from predictor import get_predictor
import torch
import time

# 1) 常规初始化
engine_path = "/root/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/encoder.plan"
face_det = get_predictor(predict_type="trt", model_path=engine_path)

# 准备一次用于捕获的输入（后续复用时形状必须一致）
batch_size = 64
# 按照 encoder.onnx 的输入定义：speech[B, T, 560], speech_lengths[B]
T = 93
speech = torch.randn(batch_size, T, 560, device="cuda", dtype=torch.float16)
speech_lengths = torch.tensor([T] * batch_size, dtype=torch.int32, device="cuda")
feed_dict = {"speech": speech, "speech_lengths": speech_lengths}

# 把本次形状设置到 context，并把数据拷到静态缓冲里
face_det.adjust_buffer(feed_dict)
for name, tensor in face_det.tensors.items():
    face_det.context.set_tensor_address(name, tensor.data_ptr())

# 2) 创建 stream，warmup，并进行图捕获
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    for _ in range(3):  # warmup
        face_det.context.execute_async_v3(stream.cuda_stream)
torch.cuda.synchronize()

from torch.cuda import nvtx
for _ in range(5):
    t1 = time.time()
    nvtx.range_push("execute")
    face_det.context.execute_async_v3(stream.cuda_stream)
    nvtx.range_pop()
    # torch.cuda.synchronize()
    stream.synchronize()
    t2 = time.time()
    print(f"execute_async_v3: {(t2 - t1) * 1000:.2f} ms")

'''
# 捕获 CUDA Graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=stream):
    face_det.context.execute_async_v3(stream.cuda_stream)

# 3) 复用函数
def run_with_graph(new_x: torch.Tensor):
    # 形状必须与捕获一致；只在原位 copy，保持指针不变
    face_det.tensors["x"][:new_x.shape[0], :new_x.shape[1], :new_x.shape[2], :new_x.shape[3]].copy_(new_x)
    t1 = time.time()
    g.replay()
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"execute_async_v3 cuda graph: {(t2 - t1) * 1000:.2f} ms")
    return face_det.tensors

# 示例推理
for i in range(10):
    new_x = torch.randn(batch_size, 3, 48, 192, device="cuda")
    # new_x = torch.randn(batch_size, 3, 48, 4096, device="cuda")
    outputs = run_with_graph(new_x)
    # print(f"Iteration {i}: output keys = {list(outputs.keys())}")
'''

# /usr/local/tensorrt/bin/trtexec --onnx=image_cls_emb.onnx --saveEngine=image_cls_emb.trt   --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 --minShapes=x:1x3x336x336 --optShapes=x:16x3x336x336 --maxShapes=x:128x3x336x336
# /usr/local/tensorrt/bin/trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet50_fp32.trt  # 6.73ms
# /usr/local/tensorrt/bin/trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet50_fp16.trt --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 # 2.14ms
# /usr/local/tensorrt/bin/trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet50_int8.trt --int8 # 1.61ms
# /usr/local/tensorrt/bin/trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet50_int8.trt --inputIOFormats=int8:chw --outputIOFormats=int8:chw --int8 # 1.04ms

# /usr/local/tensorrt/bin/trtexec --onnx=/root/repo/szfs_infer_for_test/whisper_encoder.onnx --saveEngine=whisper_encoder_fp16.trt --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 # 6.27ms
# /usr/local/tensorrt/bin/trtexec --onnx=/root/repo/szfs_infer_for_test/whisper_encoder.onnx --saveEngine=whisper_encoder_int8.trt --inputIOFormats=int8:chw --outputIOFormats=int8:chw --int8 # 32.5ms, 实际是fp32
# /usr/local/tensorrt/bin/trtexec --onnx=/root/repo/szfs_infer_for_test/whisper_encoder_quant_all.onnx --saveEngine=whisper_encoder_quant_all_int8.trt --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --best # 6ms
# /usr/local/tensorrt/bin/trtexec --onnx=/root/repo/szfs_infer_for_test/whisper_encoder_quant.onnx --saveEngine=whisper_encoder_quant_int8.trt --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --best # 5.25ms 只量化fc1 fc2 q k v out

# asr
'''
  /usr/local/tensorrt/bin/trtexec \
    --onnx=/root/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/encoder.onnx \
    --saveEngine=/root/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/encoder.plan \
    --minShapes=speech:1x1x560 \
    --optShapes=speech:16x64x560 \
    --maxShapes=speech:64x128x560 \
    --bf16 \
    --inputIOFormats=bf16:chw --outputIOFormats=bf16:chw \
    --skipInference
    # --memPoolSize=workspace:8192 \
    # --inputIOFormats=fp16:chw,int32:chw --outputIOFormats=fp16:chw,int32:chw

  /usr/local/tensorrt/bin/trtexec \
    --onnx=/root/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/encoder.onnx \
    --saveEngine=/root/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/encoder_fp32.plan \
    --minShapes=speech:1x1x560 \
    --optShapes=speech:16x64x560 \
    --maxShapes=speech:64x128x560
'''

# wan
'''
export CUDA_VISIBLE_DEVICES=0

# onnx fp16
# 1350ms, nan
trtexec --onnx=/root/models/wan_onnx/wan_model.onnx \
        --saveEngine=/root/models/wan_model_fp16_in_fp16.trt \
        --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16

# l2 relative error: 0.0832577496767044
trtexec --onnx=/root/models/wan_onnx/wan_model_bf16.onnx \
        --saveEngine=/root/models/wan_model.trt --bf16 \
        --precisionConstraints=obey \
        --layerPrecisions="*LayerNormalization:fp32,*Pow:fp32,*ReduceMean:fp32"

# 2549ms, l2 relative error: 0.08800437301397324
trtexec --onnx=/root/models/wan_onnx/wan_model.onnx \
        --saveEngine=/root/models/wan_model_bf16.trt --bf16

# 1358ms, nan
trtexec --onnx=/root/models/wan_onnx/wan_model.onnx \
        --saveEngine=/root/models/wan_model_fp16.trt --fp16

# 2250ms, l2 relative error: 0.23430126905441284
trtexec --onnx=/root/models/wan_onnx_quant/wan_model.onnx \
        --saveEngine=/root/models/wan_model_quant.trt --bf16 --int8

trtexec --onnx=/root/models/wan_onnx_fp8/wan_model.onnx \
        --saveEngine=/root/models/wan_model_fp8.trt --bf16 --fp8

ncu -o wan_open_int8 --set full --cache-control none --clock-control none --kernel-name sm80_xmma_gemm_i8f32_i8i32_f32_tn_n_tilesize128x128x64_stage3_warpsize2x2x1_tensor16x8x32_fused --launch-skip 843 --launch-count 1 "python3" test_wan_trt.py
ncu -o wan_open_bf16 --set full --cache-control none --clock-control none --kernel-name sm80_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x32_stage4_warpsize2x2x1_tensor16x8x16_fused --launch-skip 210 --launch-count 1 "python3" test_wan_trt.py

# 1150ms, l2 relative error: 1.0253195762634277
trtexec --onnx=/root/models/wan_onnx_quant/wan_model.onnx \
        --saveEngine=/root/models/wan_model_best.trt --best


'''