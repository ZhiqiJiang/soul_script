#!/usr/bin/env python3
"""Benchmark PyTorch FP16/BF16 matmul TFLOPS vs W7900D theoretical peak.

Usage:
  conda activate vllm-rocm
  export HIP_VISIBLE_DEVICES=1
  rocm-smi -d 1 --setperflevel high
  python /root/jiangzhiqi/repo/vllm/bench_fp16_peak_pytorch.py
  # slower layout (B^T):
  python /root/jiangzhiqi/repo/vllm/bench_fp16_peak_pytorch.py --transpose-b
"""

from __future__ import annotations

import argparse

import torch

# W7900D: 96 CU * 64 ALU = 6144 FP32 ALUs; matrix FP16 = 2x FP32
NUM_ALU = 6144
FP16_OVER_FP32 = 2
DEFAULT_PEAK_MHZ = 1760.0


def peak_tflops(mhz: float, dtype_name: str) -> float:
    fp32 = NUM_ALU * 4 * (mhz / 1000.0) / 1000.0  # TFLOPS
    if dtype_name in ("fp16", "bf16"):
        return fp32 * FP16_OVER_FP32
    return fp32


def bench_matmul(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    transpose_b: bool = True,
    warmup: int = 20,
    iters: int = 100,
) -> tuple[float, float]:
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    if transpose_b:
        b = torch.randn(n, k, device="cuda", dtype=dtype).t()
    else:
        b = torch.randn(k, n, device="cuda", dtype=dtype)

    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    torch.cuda.synchronize()
    _ = c  # keep last result live

    ms = start.elapsed_time(end) / iters
    tflops = 2.0 * m * n * k / (ms * 1e-3) / 1e12
    return ms, tflops


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peak-mhz", type=float, default=DEFAULT_PEAK_MHZ)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument(
        "--transpose-b",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use B^T (TN-like). Default is NN (no transpose), which is faster on W7900D.",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device not available")

    device = torch.cuda.get_device_name(0)
    peak16 = peak_tflops(args.peak_mhz, "fp16")
    transpose_b = args.transpose_b

    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"device={device}")
    print(f"peak_mhz={args.peak_mhz:.1f} FP16_peak={peak16:.3f} TFLOPS")
    print(f"transpose_b={transpose_b} warmup={args.warmup} iters={args.iters}")
    print()

    dtypes = [
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ]
    squares = [2048, 3072, 3840, 4096, 5120, 6144, 7168, 7680, 8192, 10240, 12288]
    rects = [
        (8192, 8192, 1024),
        (8192, 8192, 2048),
        (8192, 8192, 4096),
        (6144, 12288, 4096),
        (12288, 6144, 4096),
        (4096, 4096, 8192),
        (4096, 4096, 16384),
        (16384, 16384, 4096),
    ]

    rows: list[tuple[float, str, int, int, int, float, float]] = []

    for name, dtype in dtypes:
        for s in squares:
            try:
                ms, tflops = bench_matmul(
                    s, s, s, dtype,
                    transpose_b=transpose_b,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                util = 100.0 * tflops / peak16
                rows.append((util, name, s, s, s, ms, tflops))
                print(f"{name} {s}x{s}x{s}: {tflops:7.2f} TFLOPS  {util:5.1f}%  ({ms:.3f} ms)")
            except Exception as e:
                print(f"{name} {s}x{s}x{s}: FAIL {e}")
                torch.cuda.empty_cache()

    print("\n=== rectangular fp16 ===")
    for m, n, k in rects:
        try:
            ms, tflops = bench_matmul(
                m, n, k, torch.float16,
                transpose_b=transpose_b,
                warmup=args.warmup,
                iters=args.iters,
            )
            util = 100.0 * tflops / peak16
            rows.append((util, "fp16", m, n, k, ms, tflops))
            print(f"fp16 {m}x{n}x{k}: {tflops:7.2f} TFLOPS  {util:5.1f}%  ({ms:.3f} ms)")
        except Exception as e:
            print(f"fp16 {m}x{n}x{k}: FAIL {e}")
            torch.cuda.empty_cache()

    rows.sort(reverse=True)
    print("\n=== TOP 10 by util @ peak_mhz ===")
    for util, name, m, n, k, ms, tflops in rows[:10]:
        print(f"{name} {m}x{n}x{k}: {tflops:7.2f} TFLOPS  {util:5.1f}%  ({ms:.3f} ms)")

    if rows:
        best = rows[0]
        print(
            f"\nBEST: {best[1]} {best[2]}x{best[3]}x{best[4]} "
            f"= {best[6]:.2f} TFLOPS ({best[0]:.2f}% of {peak16:.3f} @ {args.peak_mhz:.0f} MHz)"
        )


if __name__ == "__main__":
    main()
