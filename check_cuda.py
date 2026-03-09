#!/usr/bin/env python
"""
Quick CUDA / PyTorch diagnostic script.

Run:
    python check_cuda.py
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import traceback


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def run_nvidia_smi() -> None:
    print_header("NVIDIA-SMI")
    exe = shutil.which("nvidia-smi")
    if not exe:
        print("nvidia-smi: not found in PATH")
        return

    print(f"path: {exe}")
    try:
        result = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,driver_version,cuda_version,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        print(f"failed to run nvidia-smi: {exc}")
        return

    if result.returncode != 0:
        print(f"nvidia-smi exit code: {result.returncode}")
        if result.stdout.strip():
            print("stdout:")
            print(result.stdout.strip())
        if result.stderr.strip():
            print("stderr:")
            print(result.stderr.strip())
        return

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        print("no GPU rows returned")
        return

    for row in lines:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 8:
            print(row)
            continue
        idx, name, driver, cuda, util, mem_used, mem_total, temp = parts[:8]
        print(
            f"GPU {idx}: {name} | driver={driver} | cuda={cuda} | "
            f"util={util}% | mem={mem_used}/{mem_total} MB | temp={temp} C"
        )


def run_torch_check() -> int:
    print_header("PYTORCH")
    try:
        import torch
    except Exception as exc:
        print("torch import: FAILED")
        print(f"error: {exc}")
        print()
        print("traceback:")
        traceback.print_exc()
        return 1

    print("torch import: OK")
    print(f"torch.__version__: {getattr(torch, '__version__', None)}")
    print(f"torch.version.cuda: {getattr(torch.version, 'cuda', None)}")

    try:
        cuda_available = bool(torch.cuda.is_available())
        print(f"torch.cuda.is_available(): {cuda_available}")
    except Exception as exc:
        print(f"torch.cuda.is_available(): FAILED -> {exc}")
        return 2

    if not cuda_available:
        print("CUDA is not available to PyTorch.")
        return 3

    try:
        device_count = int(torch.cuda.device_count())
        print(f"torch.cuda.device_count(): {device_count}")
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            print(
                f"device {idx}: {props.name} | "
                f"capability={props.major}.{props.minor} | "
                f"total_memory={props.total_memory // (1024 * 1024)} MB"
            )

        current = torch.cuda.current_device()
        print(f"torch.cuda.current_device(): {current}")
        print(f"current_device_name: {torch.cuda.get_device_name(current)}")

        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        y = x * 2
        print(f"tensor_device_test: OK -> {y}")
        return 0
    except Exception as exc:
        print(f"CUDA runtime test failed: {exc}")
        traceback.print_exc()
        return 4


def main() -> int:
    print_header("SYSTEM")
    print(f"python: {sys.version.splitlines()[0]}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.platform()}")

    run_nvidia_smi()
    code = run_torch_check()

    print_header("RESULT")
    if code == 0:
        print("CUDA looks ready for PyTorch.")
    else:
        print(f"CUDA check failed with code {code}.")
        print("If nvidia-smi works but torch import fails, the PyTorch/CUDA install is broken.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
