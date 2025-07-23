import torch
from torch.utils import benchmark
import matplotlib.pyplot as plt
import numpy as np

torch.cuda.set_device(7) 
device = torch.device("cuda:7")

torch.backends.cuda.matmul.allow_fp16_accumulation = True
typ = torch.float16 

def smooth_data(y, window_size=5):
    """使用移动平均平滑数据"""
    return np.convolve(y, np.ones(window_size)/window_size, mode='same')

k_group = [4608, 18944, 152064]
n = 3584
m_values = []
tflops_values = []

for k in k_group:
    m_values = []
    tflops_values = []
    
    for m in range(1,1024):
        a = torch.randn(m, k).type(typ).cuda()
        b = torch.randn(k, n).type(typ).cuda()

        t = benchmark.Timer(
            stmt='a @ b',
            globals={'a': a, 'b': b})

        x = t.timeit(50)
        tflops = 2 * m * n * k / x.median / 1e12
        
        m_values.append(m)
        tflops_values.append(tflops)
        print(f"m={m}, TFLOPS={tflops:.2f}")

    # 创建柱状图
    plt.figure(figsize=(15, 6))
    
    # 绘制柱状图 - 每10个点取一个样本以减少密度
    plt.bar(m_values[::10], tflops_values[::10], width=8, alpha=0.5, label='TFLOPS Samples')
    
    # 绘制平滑曲线
    smoothed_tflops = smooth_data(tflops_values)
    plt.plot(m_values, smoothed_tflops, 'r-', linewidth=2, label='Smoothed Trend')
    
    # 添加峰值参考线
    plt.axhline(y=333, color='g', linestyle='--', label='Peak TFLOPS (333)')
    
    plt.title(f'Matrix Multiplication Performance (TFLOPS) vs Matrix Size (m) k={k} n=3584')
    plt.xlabel('Matrix Size (m)')
    plt.ylabel('TFLOPS')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(f'torch_acc_bar_k={k}.png')
