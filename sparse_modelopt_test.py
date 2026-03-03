import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import modelopt.torch.sparsity as mts
from torch.sparse.semi_structured import SparseSemiStructuredTensor, SparseSemiStructuredTensorCUTLASS

# 1. Environment Setup
device = torch.device("cuda")
dtype = torch.float16

if not torch.cuda.is_available():
    print("Need NVIDIA GPU")
    sys.exit(1)

# 2. Model Setup
M, K, N = 8192, 8192, 8192 
print(f"Creating Linear Layer: [{M}x{K}] * [{K}x{N}] (dtype={dtype})")
input_tensor = torch.randn(M, K, device=device, dtype=dtype)
model = nn.Linear(K, N, bias=False).to(device).to(dtype)
out_dense_origin = model(input_tensor)

# Save original weight for dense benchmark
dense_weight = model.weight.detach().clone()

# 3. Sparsify using NVIDIA ModelOpt
# This replaces the manual topk/masking logic from the previous example
mode = "sparsegpt"
if mode == "sparse_magnitude":
    print("\n[ModelOpt] Sparsifying model using 'sparse_magnitude'...")
    mts.sparsify(model, "sparse_magnitude")
else:
    print("\n[ModelOpt] Sparsifying model using 'sparsegpt' (Data-driven sparsity)...")
    calib_dataloader = [input_tensor]
    sparsity_config = {
        "data_loader": calib_dataloader,
        "collect_func": lambda x: (x,),
    }
    mts.sparsify(model, "sparsegpt", config=sparsity_config)

print("[ModelOpt] Exporting to plain PyTorch model (applying masks)...")
# mts.export() removes the ModelOpt wrappers and bakes the mask into the weights
# The model is now a standard nn.Module with weights that have zeros in 2:4 pattern
mts.export(model)

# Verify 2:4 sparsity
weight_sparse_dense = model.weight.detach()
# Check if it follows 2:4 pattern roughly (checking zero count)
zero_count = (weight_sparse_dense == 0).sum().item()
total_elements = weight_sparse_dense.numel()
print(f"Sparsity check: {zero_count}/{total_elements} elements are zero ({zero_count/total_elements:.2%})")

# 4. Accelerate with PyTorch Semi-Structured Sparse Tensor
# ModelOpt gives us the zeroes, but PyTorch needs a special Tensor format to run fast.
print("\n[PyTorch] Converting to SparseSemiStructuredTensor for acceleration...")
try:
    # Explicit CUTLASS backend is robust
    sparse_weight = SparseSemiStructuredTensorCUTLASS.from_dense(weight_sparse_dense)
    print("Conversion successful.")
except Exception as e:
    print(f"Conversion failed: {e}")
    sys.exit(1)

# 5. Benchmark
def benchmark(func, name, iters=50):
    # Warmup
    for _ in range(5): func()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iters): func()
    torch.cuda.synchronize()
    
    avg_time = (time.time() - start) / iters * 1000
    print(f"{name}: {avg_time:.3f} ms")
    return avg_time

print("\nRunning Benchmark...")

# Dense Baseline (using the original dense weights before pruning to represent standard usage)
# Or should we use the pruned weights (but dense format)? 
# Standard comparison is usually "Dense Model" vs "Sparse Model".
# Let's use the pruned weights in dense format to be apples-to-apples on logic, 
# but usually dense model has arbitrary weights. The perf is the same for dense kernels.
model.weight.data = weight_sparse_dense
out_dense = model(input_tensor)
t_dense = benchmark(lambda: model(input_tensor), "Dense Linear (cuBLAS)")

# Sparse Inference
# Note: We must use the functional API or replace the parameter in the model (tricky).
# Simplest is Functional API.
model.weight = nn.Parameter(sparse_weight)
out_sparse = model(input_tensor)
t_sparse = benchmark(lambda: model(input_tensor), "Sparse Linear (2:4)")

print(f"\nSpeedup: {t_dense / t_sparse:.2f}x")

def diff(predict, target):
    l2_relative_error = torch.norm(predict - target, p=2) / torch.norm(target, p=2)
    rmse = torch.sqrt(torch.mean((predict - target) ** 2))
    rms = torch.sqrt(torch.mean((target) ** 2))
    similarity = torch.nn.functional.cosine_similarity(predict.reshape(-1), target.reshape(-1), dim=0)
    print(f"rmse: {rmse.item()}, rms: {rms.item()}, l2_relative_error: {l2_relative_error.item()}, similarity: {similarity}")
    return l2_relative_error, rmse, rms, similarity


diff(out_sparse.float(), out_dense_origin.float())
diff(out_sparse.float(), out_dense.float())


