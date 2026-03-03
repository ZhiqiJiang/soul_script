import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.sparse.semi_structured import SparseSemiStructuredTensor, SparseSemiStructuredTensorCUTLASS
import time
import sys
import nvtx

# Set device and dtype
device = torch.device("cuda")
dtype = torch.float16

# Check capability
if not torch.cuda.is_available():
    print("CUDA not available")
    sys.exit(1)

cap = torch.cuda.get_device_capability()
print(f"Running on {torch.cuda.get_device_name(0)} (Compute Capability {cap[0]}.{cap[1]})")
if cap[0] < 8:
    print(f"Warning: GPU capability < 8.0, 2:4 sparsity might not speed up.")

# 1. Setup Model (Single Linear Layer)
# Using large dimensions to highlight speedup (small layers are memory bound)
M, K, N = 8192, 8192, 8192 
print(f"Creating Linear Layer: [{M}x{K}] * [{K}x{N}] (dtype={dtype})")

input_tensor = torch.randn(M, K, device=device, dtype=dtype)
linear = nn.Linear(K, N, bias=False).to(device).to(dtype)
out_dense_origin = linear(input_tensor)

# 2. Sparsify Weights to 2:4 Pattern
print("Pruning weights to 2:4 pattern...")
weight = linear.weight.detach()
# Shape is (N, K). We need 2 zeros in every block of 4 along K dimension (dim=1)
# Reshape to (N, K//4, 4)
w_reshaped = weight.view(N, -1, 4)
# Find indices of the 2 smallest values in the last dimension
_, indices = torch.topk(w_reshaped.abs(), 2, dim=-1, largest=False)
# Create mask
mask = torch.ones_like(w_reshaped, dtype=torch.bool)
mask.scatter_(-1, indices, False)
# Apply mask
weight_2_4 = w_reshaped * mask
weight_2_4 = weight_2_4.view(N, K)

# Verify
assert (weight_2_4 != 0).sum() <= weight_2_4.numel() / 2
print("Sparsity verified.")

# 3. Convert to SparseSemiStructuredTensor
print("Converting to SparseSemiStructuredTensor...")
sparse_weight = None
try:
    # Try explicit CUTLASS backend
    print("Attempting conversion using CUTLASS backend...")
    sparse_weight = SparseSemiStructuredTensorCUTLASS.from_dense(weight_2_4)
    print("Conversion successful (CUTLASS backend).")
except Exception as e:
    print(f"CUTLASS conversion failed: {e}")
    try:
         # Try cuSPARSELt if available
         from torch.sparse.semi_structured import SparseSemiStructuredTensorCUSPARSELT
         print("Attempting conversion using cuSPARSELt backend...")
         sparse_weight = SparseSemiStructuredTensorCUSPARSELT.from_dense(weight_2_4)
         print("Conversion successful (cuSPARSELt backend).")
    except Exception as e2:
        import traceback
        traceback.print_exc()
        print(f"All conversions failed.")
        sys.exit(1)

# 4. Benchmark
def benchmark(func, name, iters=200):
    # Warmup
    for _ in range(10):
        func()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iters):
        func()
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iters * 1000
    print(f"{name}: {avg_time:.3f} ms")
    return avg_time

print("\nRunning Benchmark...")

# Dense Baseline
# Note: Using the sparsified weight but in dense format to be fair on values (though Dense doesn't care)
linear.weight.data = weight_2_4
dense_time = benchmark(lambda: linear(input_tensor), "Dense Linear (cuBLAS)")

# Sparse 
# Note: F.linear supports SparseSemiStructuredTensor as weight
sparse_time = benchmark(lambda: F.linear(input_tensor, sparse_weight), "Sparse Linear (2:4)")

print(f"\nSpeedup: {dense_time / sparse_time:.2f}x")

def diff(predict, target):
    l2_relative_error = torch.norm(predict - target, p=2) / torch.norm(target, p=2)
    rmse = torch.sqrt(torch.mean((predict - target) ** 2))
    rms = torch.sqrt(torch.mean((target) ** 2))
    similarity = torch.nn.functional.cosine_similarity(predict.reshape(-1), target.reshape(-1), dim=0)
    print(f"rmse: {rmse.item()}, rms: {rms.item()}, l2_relative_error: {l2_relative_error.item()}, similarity: {similarity}")
    return l2_relative_error, rmse, rms, similarity

out_dense = linear(input_tensor)
out_sparse = F.linear(input_tensor, sparse_weight)
diff(out_sparse.float(), out_dense_origin.float())
diff(out_sparse.float(), out_dense.float())