import torch
from torch.utils import benchmark
import cutlass

typ = torch.float16  #数据精度
n = 1024 * 8
a = torch.ones(n, n).type(typ).cuda()
b = torch.ones(n, n).type(typ).cuda()
c = torch.ones(n, n).type(typ).cuda()
d = torch.ones(n, n).type(typ).cuda()

t = benchmark.Timer(
      stmt='torch.mm(a, b)',
      globals={'a': a, 'b': b, 'torch': torch})

x = t.timeit(50)
print(2*n**3 / x.median /1e12)

plan = cutlass.op.Gemm(element=cutlass.DataType.f16, element_accumulator=cutlass.DataType.f16, layout=cutlass.LayoutType.RowMajor)
t = benchmark.Timer(
      stmt='plan.run(a, b, c, d)',
      globals={'a': a, 'b': b, 'c': c, 'd':d, 'plan': plan})

x = t.timeit(50)
print(2*n**3 / x.median /1e12)