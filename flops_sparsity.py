#!/usr/bin/env python3
import os
import time

import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer


def main():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return 1

    num_devices = torch.cuda.device_count()
    print(f"CUDA devices: {num_devices}")
    for i in range(num_devices):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        sm = cap[0] * 10 + cap[1]
        print(f"device {i}: {name}, capability {cap}, SM{sm}")

    device = torch.device("cuda:0")

    # Semi-structured sparsity acceleration requires A100/H100 (SM80/SM90)
    cap = torch.cuda.get_device_capability(0)
    sm = cap[0] * 10 + cap[1]

    # Model dims similar to common MLP shapes
    in_features = 10240
    out_features = 3072
    batch = 3072  # x shape will be [batch, in_features]

    linear = torch.nn.Linear(in_features, out_features, bias=False).to(device).half().eval()

    # Create input
    x = torch.randn(batch, in_features, device=device, dtype=torch.float16)

    # Warmup dense
    with torch.inference_mode():
        for _ in range(10):
            _ = linear(x)
        torch.cuda.synchronize()

    # Measure dense
    with torch.inference_mode():
        dense_t = Timer(stmt="linear(x)", globals={"linear": linear, "x": x}).blocked_autorange().median * 1e3

    # Mask the weight to be 2:4 sparse in a trivial pattern to meet constraints
    # This is only to demonstrate acceleration; real usage should prune by magnitude
    weight = linear.weight.data
    # Shape [out_features, in_features], enforce 2 zeros per group of 4 along last dim
    mask = torch.zeros_like(weight, dtype=torch.bool, device=device)
    # Pattern: keep last two in every group of 4
    keep = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.bool)
    reps = (weight.shape[1] // 4)
    keep_row = keep.repeat(reps)[: weight.shape[1]]
    mask[:, :] = keep_row
    linear.weight = torch.nn.Parameter(weight * mask)

    # Convert to SparseSemiStructuredTensor for acceleration
    SparseSemiStructuredTensor._FORCE_CUTLASS = True  # prefer cuSPARSELt when available
    linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))

    # Validate correctness roughly
    with torch.inference_mode():
        y_dense = (mask * weight).to(torch.float16)  # ensure same dtype
        y_ref = torch.matmul(x, y_dense.t())
        y_sparse = linear(x)
        max_abs = (y_sparse - y_ref).abs().max().item()
        print(f"Max abs diff (sparse vs dense masked): {max_abs:.3e}")

    # Warmup sparse
    with torch.inference_mode():
        for _ in range(10):
            _ = linear(x)
        torch.cuda.synchronize()

    # Measure sparse
    with torch.inference_mode():
        sparse_t = Timer(stmt="linear(x)", globals={"linear": linear, "x": x}).blocked_autorange().median * 1e3

    print(f"Dense: {dense_t:.3f} ms | Sparse: {sparse_t:.3f} ms | Speedup: {dense_t / sparse_t:.3f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 