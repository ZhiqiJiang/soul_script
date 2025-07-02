import torch
from torch.utils import benchmark
import transformer_engine.pytorch as te

n = 1024 * 8
a = torch.randn(n, n, device='cuda', dtype=torch.float32)
b = torch.randn(n, n, device='cuda', dtype=torch.float32)

# 转换为 FP8
a_fp8, _ = te.fp8_autocast.convert_to_fp8(a, fp8_format=te.fp8_format.E4M3)
b_fp8, _ = te.fp8_autocast.convert_to_fp8(b, fp8_format=te.fp8_format.E4M3)

t = benchmark.Timer(
      stmt='te.matmul(a_fp8, b_fp8)',
      globals={'a_fp8': a_fp8, 'b_fp8': b_fp8, 'te': te})

x = t.timeit(50)
print(2*n**3 / x.median /1e12)