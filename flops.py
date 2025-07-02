import torch
from torch.utils import benchmark
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_fp16_accumulation = True
typ = torch.float16
m = 600
k = 3584
n = 37888
a = torch.randn(m, k).type(typ).cuda()
b = torch.randn(k, n).type(typ).cuda()

t = benchmark.Timer(
      stmt='a @ b',
      globals={'a': a, 'b': b})

x = t.timeit(50)
print(f"{2*m*n*k / x.median /1e12:.2f} TFLOPS")
print(f"{x.median * 1000:.2f} ms")

# t = benchmark.Timer(
#       stmt='F.linear(a, b.T)',
#       globals={'a': a, 'b': b, 'F': F})

# x = t.timeit(50)
# print(f"{2*m*n*k / x.median /1e12:.2f} TFLOPS")
# print(f"{x.median * 1000:.2f} ms")


# x shape: torch.Size([600, 3584]), weight shape: torch.Size([37888, 3584]), bias shape: None
# x shape: torch.Size([600, 18944]), weight shape: torch.Size([3584, 18944]), bias shape: None
# x shape: torch.Size([600, 3584]), weight shape: torch.Size([4608, 3584]), bias shape: torch.Size([4608])