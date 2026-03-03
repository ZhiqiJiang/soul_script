"""
Universal Model Sparsification Acceleration Framework

This module provides a simple API to accelerate any PyTorch model using 2:4 sparsity.

Usage:
    from sparse_acceleration import enable_sparse_acceleration, SparseOptions
    
    model = YourModel()
    model = enable_sparse_acceleration(model, calibration_data=[sample_input])
    output = model(input)  # Automatically uses sparse acceleration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import modelopt.torch.sparsity as mts
from torch.sparse.semi_structured import SparseSemiStructuredTensorCUTLASS


@dataclass(frozen=True)
class SparseOptions:
    """Configuration options for sparse acceleration."""
    
    mode: Literal["sparse_magnitude", "sparsegpt"] = "sparse_magnitude"
    
    calibration_data: Optional[List[torch.Tensor]] = None
    
    materialize_on_wrap: bool = True
    
    keep_original_weights: bool = False
    
    verbose: bool = True


class SparseLinear(nn.Module):
    """Drop-in replacement for nn.Linear with 2:4 sparse acceleration."""

    def __init__(self, linear: nn.Linear, *, options: SparseOptions):
        super().__init__()
        
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(linear)}")
        
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        self.options = options
        
        self.original_linear: Optional[nn.Linear] = None
        self.bias: Optional[nn.Parameter] = None
        
        if options.keep_original_weights:
            self.original_linear = linear
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None
            self._original_weight_cpu = linear.weight.detach().to(device="cpu").contiguous()
        
        self._sparse_weight: Optional[SparseSemiStructuredTensorCUTLASS] = None
        self._dense_sparse_weight: Optional[nn.Parameter] = None
        self._is_materialized = False
        
        self._weight = nn.Parameter(linear.weight.detach().clone())
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, *, options: SparseOptions) -> "SparseLinear":
        return cls(linear, options=options)
    
    def sparsify(self, calibration_data: Optional[List[torch.Tensor]] = None):
        """Apply sparsification to the weight."""
        if self.options.verbose:
            print(f"  Sparsifying Linear({self.in_features}, {self.out_features})")
        
        temp_linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        temp_linear.weight.data = self._weight.data
        if self.bias is not None:
            temp_linear.bias.data = self.bias.data
        
        if self.options.mode == "sparsegpt":
            if calibration_data is None:
                calibration_data = self.options.calibration_data
            if calibration_data is None:
                raise ValueError("sparsegpt mode requires calibration_data")
            
            config = {
                "data_loader": calibration_data,
                "collect_func": lambda x: (x,),
            }
            mts.sparsify(temp_linear, "sparsegpt", config=config)
        elif self.options.mode == "sparse_magnitude":
            mts.sparsify(temp_linear, "sparse_magnitude")
        else:
            raise ValueError(f"Unsupported sparsity mode: {self.options.mode}")
        
        mts.export(temp_linear)
        
        self._weight.data = temp_linear.weight.data
        
        if self.options.verbose:
            zero_count = (self._weight.data == 0).sum().item()
            total = self._weight.data.numel()
            print(f"    Sparsity: {zero_count/total:.2%} ({zero_count}/{total} zeros)")
    
    def materialize_sparse_tensor(self, device: torch.device):
        """Convert the sparse weight to SparseSemiStructuredTensor for acceleration."""
        if self._is_materialized and self._dense_sparse_weight is not None:
            return
        
        if self.options.verbose:
            print(f"  Materializing sparse tensor for Linear({self.in_features}, {self.out_features})")
        
        try:
            sparse_tensor = SparseSemiStructuredTensorCUTLASS.from_dense(self._weight.data)
            self._sparse_weight = sparse_tensor
            self._dense_sparse_weight = nn.Parameter(sparse_tensor, requires_grad=False)
            self._is_materialized = True
            
            if self.options.verbose:
                print("    ✓ Successfully converted to sparse tensor")
        except Exception as e:
            if self.options.verbose:
                print(f"    ✗ Failed to convert: {e}")
            self._dense_sparse_weight = self._weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            if self.original_linear is not None:
                return self.original_linear(x)
            return F.linear(x, self._weight, self.bias)
        
        if not self._is_materialized:
            self.materialize_sparse_tensor(x.device)
        
        return F.linear(x, self._dense_sparse_weight, self.bias)


def enable_sparse_acceleration(
    model: nn.Module,
    *,
    options: SparseOptions = SparseOptions(),
    calibration_data: Optional[List[torch.Tensor]] = None,
    module_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    inplace: bool = True,
) -> nn.Module:
    """
    Enable 2:4 sparse acceleration for a PyTorch model.
    
    This function replaces nn.Linear modules with SparseLinear modules that
    automatically use sparse tensor acceleration when running on CUDA.
    
    Args:
        model: Any PyTorch model (nn.Module)
        options: SparseOptions controlling sparsification behavior
        calibration_data: Calibration data for sparsegpt mode
        module_filter: Optional function (name, module) -> bool to select which
                      Linear layers to sparsify. If None, sparsifies all Linear layers.
        inplace: If True, modifies model in-place. If False, creates a deep copy.
    
    Returns:
        The model with sparse acceleration enabled
    
    Example:
        >>> from sparse_acceleration import enable_sparse_acceleration, SparseOptions
        >>> 
        >>> model = MyModel()
        >>> sample_input = torch.randn(1, 3, 224, 224).cuda()
        >>> model = enable_sparse_acceleration(
        ...     model,
        ...     options=SparseOptions(mode="sparse_magnitude"),
        ...     calibration_data=[sample_input]
        ... )
        >>> output = model(sample_input)  # Uses sparse acceleration!
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    def should_sparsify(name: str, m: nn.Module) -> bool:
        if not isinstance(m, nn.Linear):
            return False
        if module_filter is None:
            return True
        return bool(module_filter(name, m))
    
    sparse_modules: List[SparseLinear] = []
    
    if should_sparsify("", model):
        if options.verbose:
            print(f"\n{'='*80}")
            print(f"Found 1 Linear layer to sparsify (top-level module)")
            print(f"{'='*80}")
        
        sparse_mod = SparseLinear.from_linear(model, options=options)
        sparse_modules.append(sparse_mod)
        
        if options.verbose:
            print("\n[Step 1/2] Applying sparsification...")
        
        for sparse_mod in sparse_modules:
            sparse_mod.sparsify(calibration_data)
        
        if options.materialize_on_wrap:
            if options.verbose:
                print("\n[Step 2/2] Materializing sparse tensors...")
            
            device = sparse_modules[0]._weight.device
            for sparse_mod in sparse_modules:
                sparse_mod.materialize_sparse_tensor(device)
        
        if options.verbose:
            print(f"\n{'='*80}")
            print("✓ Sparse acceleration enabled successfully!")
            print(f"{'='*80}\n")
        
        return sparse_modules[0]
    
    def _recurse(prefix: str, parent: nn.Module) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if should_sparsify(full_name, child):
                sparse_mod = SparseLinear.from_linear(child, options=options)
                setattr(parent, child_name, sparse_mod)
                sparse_modules.append(sparse_mod)
            else:
                _recurse(full_name, child)
    
    _recurse("", model)
    
    if options.verbose:
        print(f"\n{'='*80}")
        print(f"Found {len(sparse_modules)} Linear layers to sparsify")
        print(f"{'='*80}")
    
    if sparse_modules:
        if options.verbose:
            print("\n[Step 1/2] Applying sparsification...")
        
        for sparse_mod in sparse_modules:
            sparse_mod.sparsify(calibration_data)
        
        if options.materialize_on_wrap:
            if options.verbose:
                print("\n[Step 2/2] Materializing sparse tensors...")
            
            device = next(model.parameters()).device if list(model.parameters()) else torch.device("cuda")
            for sparse_mod in sparse_modules:
                sparse_mod.materialize_sparse_tensor(device)
    
    if options.verbose:
        print(f"\n{'='*80}")
        print("✓ Sparse acceleration enabled successfully!")
        print(f"{'='*80}\n")
    
    return model

