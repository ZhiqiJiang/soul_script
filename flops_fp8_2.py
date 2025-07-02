import torch
import torch.nn.functional as F
print(torch.cuda.get_device_capability())
# 检查GPU是否支持FP8
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:  # Hopper架构及以上
    # 设置设备
    device = torch.device("cuda")
    
    # 创建示例矩阵（使用BF16作为中间格式）
    a = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    
    # 创建FP8类型配置
    fp8_e4m3 = torch.float8_e4m3fn  # E4M3格式，适合较大范围
    fp8_e5m2 = torch.float8_e5m2    # E5M2格式，适合高精度
    
    # 转换为FP8
    a_fp8 = a.to(fp8_e4m3)
    b_fp8 = b.to(fp8_e4m3)
    
    # 使用FP8进行矩阵乘法
    # 注意：PyTorch目前推荐使用e4m3格式进行矩阵乘法
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float8_e4m3):
        c_fp8 = torch.matmul(a_fp8, b_fp8)
    
    # 将结果转回BF16或FP32进行后续计算
    c_bf16 = c_fp8.to(torch.bfloat16)
    
    # 验证结果（与FP32结果对比）
    a_fp32 = a.to(torch.float32)
    b_fp32 = b.to(torch.float32)
    c_fp32 = torch.matmul(a_fp32, b_fp32)
    
    # 计算误差
    max_error = torch.max(torch.abs(c_bf16.to(torch.float32) - c_fp32)).item()
    print(f"最大误差: {max_error}")
    
else:
    print("当前GPU不支持FP8计算，需要NVIDIA Hopper架构或更新的GPU。")