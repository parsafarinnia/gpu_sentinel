
# GPU Sentinel

A tiny, self-contained GPU “sentry” that monitors all visible NVIDIA GPUs and opportunistically runs a local workload when a GPU is idle. If another (foreign) process starts using a GPU, the sentry stops its local workload on that GPU immediately.

- **Idle GPU →** start a stressor on that GPU  
- **Busy GPU →** stop our stressor (if running) and yield the device  
- **Graceful shutdown** on `SIGINT` / `SIGTERM`

This is useful for:
- Synthetic load testing (memory, SM activity) on free GPUs
- Burn-in / thermal testing during cluster bring-up
- Reserving GPUs without interfering with other users’ jobs

> ⚠️ **Safety**: The stressor only terminates its *own* process. It does **not** kill or signal foreign processes. It is designed to be a polite background occupant.

---

## Features

- Per-GPU child process that binds to one CUDA device  
- Allocates ~88% of currently free memory (configurable in code)  
- Lightweight arithmetic loop to keep SMs active  
- Polls `nvidia-smi` for foreign processes and reacts within ~1s  
- Clean shutdown; releases CUDA contexts on exit  

---

## Requirements

- NVIDIA driver + CUDA-capable GPU(s)  
- Python 3.9+  
- PyTorch with CUDA build (matching your driver/CUDA runtime)  
- `nvidia-smi` available on `PATH`  

Minimal `requirements.txt`:


torch  # install the right CUDA build for your system

````

Install PyTorch per the official selector: <https://pytorch.org/get-started/locally/>

---

## Quick Start

```bash
# (Optional) create a fresh venv/conda env
python -m venv .venv && source .venv/bin/activate
# or: conda create -n gpu-sentinel python=3.11 -y && conda activate gpu-sentinel

pip install torch  # pick the correct CUDA wheel for your system

# Run
python gpu_sentinel.py
````

You should see logs like:

```
[Supervisor] Watching GPUs...
[GPU 0] Free. Launching stressor.
[GPU 0] Attempting to allocate 59.13 GB
[GPU 0] Allocated 59.13 GB in 111 chunks
[GPU 1] In use. No stressor running.
...
```

When another job starts on GPU 0:

```
[GPU 0] In use by other processes. Killing stressor.
```

Stop the sentinel with `Ctrl+C` (SIGINT).

---

## How It Works

1. **Discovery:** `torch.cuda.device_count()` enumerates visible devices.
2. **Foreign PIDs:** `nvidia-smi --query-compute-apps=gpu_uuid,pid` lists compute processes per GPU UUID. We map GPU UUID → device index.
3. **Policy:** If a GPU has no foreign PIDs, we ensure a sentry **stressor** is running there; otherwise, we terminate ours.
4. **Stressor:** The child process:

   * Measures **current free memory** with `torch.cuda.mem_get_info`.
   * Allocates \~88% of free memory in 512MB chunks.
   * Runs a small compute loop to keep the GPU active.
   * Sleeps briefly to remain responsive to termination.

---

## Customizing the Workload

The compute loop lives in `stress_gpu()`:

```python
# --- Workload loop (replace if desired) ---
while True:
    for i in range(len(chunks)):
        r = torch.randn_like(chunks[i])
        chunks[i] = chunks[i] - chunks[i] * 0.0001 + chunks[i] * 0.0001 * r
    time.sleep(0.1)
```

You can replace that with, for example, a batched matrix multiply:

```python
# Example: matmul workload (toy)
hidden = 4096
A = torch.randn(hidden, hidden, device=f"cuda:{gpu_id}")
B = torch.randn(hidden, hidden, device=f"cuda:{gpu_id}")
while True:
    C = A @ B  # GEMM
    torch.cuda.synchronize()
    time.sleep(0.05)
```

Tips:

* Keep iterations short to ensure quick termination.
* Avoid unbounded tensor growth; reuse preallocated buffers.
* Consider inserting a small `sleep` to reduce log spam and improve responsiveness.

---

## Operational Notes

* **Visibility:** The sentinel only sees GPUs in the current environment (e.g., respect `CUDA_VISIBLE_DEVICES`).
* **Memory Headroom:** The 88% target is a heuristic to avoid OOM. Tweak `alloc_bytes_target = int(free_mem_bytes * 0.88)`.
* **Chunk Size:** 512MB chunks improve allocation stability. Adjust `chunk_size` if you see fragmentation.
* **Polling Interval:** Change `supervisor_loop(poll_interval_s=1.0)` to tune reactivity vs. overhead.

---

## Troubleshooting

* **`CUDA error: out of memory`:** Lower the allocation fraction or chunk size.
* **No GPUs detected:** Check `nvidia-smi`, drivers, and that your environment exposes the devices.
* **Slow shutdown:** Ensure your workload loop includes brief sleeps and avoids extremely long kernels.

---

## Repository Layout (suggested)

```
gpu-sentinel/
├─ gpu_sentinel.py
├─ README.md
├─ requirements.txt
└─ LICENSE  (MIT, suggested)
```

Example `LICENSE` (MIT):

```text
MIT License
Copyright (c) 2025 …

Permission is hereby granted, free of charge, to any person obtaining a copy
...
```

---

## FAQ

**Q: Does this kill other users’ jobs?**
A: No. It only terminates the local stressor processes it launched.

**Q: Can I restrict it to a subset of GPUs?**
A: Yes—set `CUDA_VISIBLE_DEVICES` before launching, or add a simple whitelist in `supervisor_loop()`.

**Q: Can I make it “reserve” memory without doing compute?**
A: Yes—comment out the compute loop and just keep the allocated tensors alive.

---

## Credits

Authored by Parsa Farinneya (and contributors). PRs welcome.

```

```
