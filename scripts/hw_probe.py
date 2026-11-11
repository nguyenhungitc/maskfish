#!/usr/bin/env python3
"""Hardware probe for scripts/pipeline.sh and scripts/autotune.sh: prints shell assignments.

  eval "$(python3 scripts/hw_probe.py [DIR])"

  HW_CORES        logical CPUs           HW_PHYS        physical cores
  HW_RAM_MB       total RAM              HW_FREE_MB     available RAM
  HW_DISK_MB      free disk space of DIR (default: the current directory)
  HW_GPUS         CUDA devices (0 without torch or CUDA)
  HW_GPU_NAME     name of the first one  HW_VRAM_MB     its memory
  HW_BATCH        a training batch for that memory (the trainer's 16384 at 6 GB or more, halved below)
"""
import os
import shutil
import subprocess
import sys


def phys_cores():
    try:
        seen = set()
        core = phys = None
        with open("/proc/cpuinfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "physical id":
                    phys = v
                elif k == "core id":
                    core = v
                elif not k:
                    if core is not None:
                        seen.add((phys, core))
                    core = phys = None
        if core is not None:
            seen.add((phys, core))
        if seen:
            return len(seen)
    except OSError:
        pass
    return os.cpu_count() or 1


def meminfo():
    total = free = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k == "MemTotal":
                    total = int(v.split()[0]) // 1024
                elif k == "MemAvailable":
                    free = int(v.split()[0]) // 1024
    except OSError:
        pass
    if not total:  # macOS
        try:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) >> 20
            free = total // 2
        except (OSError, ValueError):
            total = free = 4096
    return total, free


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        if out:
            name, mem = out[0].rsplit(",", 1)
            return len(out), name.strip(), int(float(mem))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return torch.cuda.device_count(), p.name, p.total_memory >> 20
    except Exception:
        pass
    return 0, "", 0


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    total, free = meminfo()
    n, name, vram = gpu()
    batch = 16384
    if vram:
        while batch > 1024 and vram < 6144 * batch // 16384:
            batch //= 2
    vals = {
        "HW_CORES": os.cpu_count() or 1, "HW_PHYS": phys_cores(), "HW_RAM_MB": total, "HW_FREE_MB": free,
        "HW_DISK_MB": shutil.disk_usage(d).free >> 20, "HW_GPUS": n, "HW_GPU_NAME": name, "HW_VRAM_MB": vram,
        "HW_BATCH": batch,
    }
    for k, v in vals.items():
        print(f"{k}='{str(v).replace(chr(39), '')}'")


if __name__ == "__main__":
    main()
