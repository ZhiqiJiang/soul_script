"""
Test examples for the Sparse Acceleration Framework

This file demonstrates how to use the sparse acceleration API.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from sparse_acceleration import (
    enable_sparse_acceleration,
    SparseOptions
)

def benchmark_model(
    model: nn.Module,
    input_data: Union[torch.Tensor, tuple, dict],
    iters: int = 50,
    warmup: int = 5,
    name: str = "Model"
) -> float:
    """
    Benchmark a model's inference speed.
    
    Args:
        model: The model to benchmark
        input_data: Input tensor or tuple/dict of inputs
        iters: Number of iterations for timing
        warmup: Number of warmup iterations
        name: Name for printing
    
    Returns:
        Average inference time in milliseconds
    """
    def run_inference():
        with torch.no_grad():
            if isinstance(input_data, dict):
                return model(**input_data)
            elif isinstance(input_data, tuple):
                return model(*input_data)
            else:
                return model(input_data)
    
    for _ in range(warmup):
        run_inference()
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    import time
    start = time.time()
    for _ in range(iters):
        run_inference()
    torch.cuda.synchronize()
    
    avg_time = (time.time() - start) / iters * 1000
    print(f"{name}: {avg_time:.3f} ms")
    return avg_time

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Sparse Acceleration Framework - Examples")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Sparse acceleration requires NVIDIA GPU.")
        import sys
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Example 1: Simple Linear Model")
    print("="*80)
    
    model1 = nn.Linear(8192, 8192, bias=False).cuda().half()
    input1 = torch.randn(2048, 8192, device="cuda", dtype=torch.float16)
    
    print("\n[Dense Model Benchmark]")
    dense_time = benchmark_model(model1, input1, name="Dense")
    
    print("\n[Enabling Sparse Acceleration...]")
    model1_sparse = enable_sparse_acceleration(
        model1,
        options=SparseOptions(mode="sparse_magnitude", verbose=True),
        calibration_data=[input1]
    )
    
    print("\n[Sparse Model Benchmark]")
    sparse_time = benchmark_model(model1_sparse, input1, name="Sparse")
    
    print(f"\n⚡ Speedup: {dense_time/sparse_time:.2f}x")
    
    print("\n" + "="*80)
    print("Example 2: Multi-Layer Model (3 lines of code!)")
    print("="*80)
    
    model2 = nn.Sequential(
        nn.Linear(8192, 8192, bias=False),
        nn.ReLU(),
        nn.Linear(8192, 8192, bias=False),
        nn.ReLU(),
        nn.Linear(8192, 8192, bias=False)
    ).cuda().half()
    
    input2 = torch.randn(2048, 8192, device="cuda", dtype=torch.float16)
    
    print("\n[Dense Model Benchmark]")
    dense_time2 = benchmark_model(model2, input2, name="Dense")
    
    print("\n[Enabling Sparse Acceleration...]")
    model2 = enable_sparse_acceleration(
        model2,
        options=SparseOptions(mode="sparse_magnitude"),
        calibration_data=[input2]
    )
    
    print("\n[Sparse Model Benchmark]")
    sparse_time2 = benchmark_model(model2, input2, name="Sparse")
    
    print(f"\n⚡ Speedup: {dense_time2/sparse_time2:.2f}x")
    
    print("\n" + "="*80)
    print("Example 3: Custom Model with Selective Sparsification")
    print("="*80)
    
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4096, 8192, bias=False)
            self.fc2 = nn.Linear(8192, 4096, bias=False)
            self.fc3 = nn.Linear(4096, 4096, bias=False)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model3 = CustomModel().cuda().half()
    input3 = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
    
    print("\n[Dense Model Benchmark]")
    dense_time3 = benchmark_model(model3, input3, name="Dense")
    
    def filter_large_layers(name: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear) and module.out_features >= 4096
    
    print("\n[Enabling Sparse Acceleration (selective)...]")
    model3 = enable_sparse_acceleration(
        model3,
        options=SparseOptions(mode="sparse_magnitude"),
        calibration_data=[input3],
        module_filter=filter_large_layers
    )
    
    print("\n[Sparse Model Benchmark]")
    sparse_time3 = benchmark_model(model3, input3, name="Sparse")
    
    print(f"\n⚡ Speedup: {dense_time3/sparse_time3:.2f}x")
    
    print("\n" + "="*80)
    print("✅ All examples completed successfully!")
    print("="*80 + "\n")
