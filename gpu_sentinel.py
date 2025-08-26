#!/usr/bin/env python3
"""
gpu_sentinel.py

A simple GPU "sentry" that:
- Continuously scans all visible NVIDIA GPUs.
- If a GPU is idle (no foreign processes), launches a local stressor process on it
  that allocates ~88% of free memory and performs a lightweight compute loop.
- If a foreign process appears on a GPU, it terminates the local stressor there.
- Shuts down cleanly on SIGINT/SIGTERM.

Notes:
- "Foreign processes" means any process other than the stressor we launched.
- The stressor loop is intentionally simple; replace it with your own workload
  (e.g., batched matmuls) if desired—see comments in stress_gpu().
"""

import os
import time
import signal
import subprocess
import multiprocessing as mp
import torch


# --------------------------------------------
# GPU Worker
# --------------------------------------------
def stress_gpu(gpu_id: int) -> None:
    """
    Occupies ~88% of currently free memory on `gpu_id`, then runs a tiny compute loop.

    Implementation details:
    - Uses torch.cuda.mem_get_info to measure *current* free bytes.
    - Allocates in 512MB chunks to avoid a single huge allocation failure.
    - Keeps tensors alive in a Python list (`chunks`) to prevent GC.
    - The inner loop does a tiny arithmetic jitter on the whole buffer to create GPU load.

    Replace the inner loop with your own workload if you want (e.g., matrix
    multiplications) but keep in mind:
      * Long-running kernels will reduce reactivity to shutdown signals.
      * Consider inserting small sleeps to yield the scheduler.

    Args:
        gpu_id: Zero-based CUDA device index.
    """
    # Bind this process to a single CUDA device
    torch.cuda.set_device(gpu_id)

    # How much memory is free *right now*?
    free_mem_bytes = torch.cuda.mem_get_info(gpu_id)[0]
    # Target allocation is a fraction of free memory to avoid OOM/TRAP
    alloc_bytes_target = int(free_mem_bytes * 0.88)  # 88% of *free* capacity
    # Number of float32 elements we can fit in that budget
    target_elements = alloc_bytes_target // 4

    print(f"[GPU {gpu_id}] Attempting to allocate {alloc_bytes_target / 1e9:.2f} GB")

    # Allocate in chunks to improve stability and reduce fragmentation pressure
    chunks = []
    chunk_size = 512 * 1024 * 1024  # 512 MB in bytes
    floats_per_chunk = chunk_size // 4
    allocated_elements = 0

    try:
        # Repeatedly allocate until we reach our target or hit allocator errors
        while allocated_elements < target_elements:
            n = min(floats_per_chunk, target_elements - allocated_elements)
            # Keep allocations device-local and contiguous
            chunk = torch.ones(n, dtype=torch.float32, device=f"cuda:{gpu_id}")
            chunks.append(chunk)
            allocated_elements += chunk.numel()
    except RuntimeError as e:
        # Expected when you slightly overshoot available memory
        print(f"[GPU {gpu_id}] Allocation stopped: {e}")

    print(
        f"[GPU {gpu_id}] Allocated {allocated_elements * 4 / 1e9:.2f} GB "
        f"in {len(chunks)} chunks"
    )

    # --- Workload loop -------------------------------------------------------
    # This is where you can swap in your own kernel(s). For example:
    #   - batched GEMMs (matmul)
    #   - convolution sweeps
    #   - synthetic bandwidth tests
    #
    # Keep iterations short-ish so that termination remains responsive.
    # -------------------------------------------------------------------------
    try:
        while True:
            # Tiny arithmetic "jitter" to keep SMs warm without being too heavy
            for i in range(len(chunks)):
                # Avoid creating new tensors of different sizes each time
                r = torch.randn_like(chunks[i])
                chunks[i] = chunks[i] - chunks[i] * 0.0001 + chunks[i] * 0.0001 * r
            time.sleep(0.1)  # yield a bit to let the supervisor react
    except KeyboardInterrupt:
        # Normal path during shutdown
        pass


# --------------------------------------------
# GPU Usage Checker
# --------------------------------------------
def get_gpu_processes() -> dict[int, list[int]]:
    """
    Returns a mapping {gpu_id: [pid, ...]} of *compute* processes currently on each GPU.

    We use `nvidia-smi` to query:
      - gpu_uuid,pid for all compute apps
      - a uuid -> index map for visible GPUs

    Returns:
        usage: dict where keys are GPU indices [0..N-1] and values are lists of PIDs.
    """
    # Query all compute processes (one row per PID per GPU UUID)
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
    ).decode().strip()

    # Initialize empty usage map for each visible device
    num_devices = torch.cuda.device_count()
    usage = {i: [] for i in range(num_devices)}

    # Build a UUID -> gpu_id map
    uuid_map: dict[str, int] = {}
    for i in range(num_devices):
        uuid = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", f"--id={i}"]
        ).decode().strip()
        uuid_map[uuid] = i

    # Fill usage table
    if output:
        for line in output.splitlines():
            uuid, pid = [x.strip() for x in line.split(",")]
            gpu_id = uuid_map.get(uuid)
            if gpu_id is not None:
                usage[gpu_id].append(int(pid))

    return usage


# --------------------------------------------
# Supervisor
# --------------------------------------------
def supervisor_loop(poll_interval_s: float = 1.0) -> None:
    """
    Main supervision loop. Launches or kills per-GPU stressor processes.

    Behavior:
    - For each GPU:
        - If there are no foreign PIDs:
            - Ensure a stressor process for that GPU is running (start if needed).
        - If there are foreign PIDs:
            - Ensure *no* stressor is running there (terminate ours if present).

    Args:
        poll_interval_s: seconds to sleep between scans.
    """
    processes: dict[int, mp.Process] = {}  # {gpu_id: process}
    stressor_pids: dict[int, int] = {}     # {gpu_id: stressor_pid}

    print("[Supervisor] Watching GPUs...")

    while True:
        gpu_usage = get_gpu_processes()

        for gpu_id, pids in gpu_usage.items():
            # Our own stressor's PID (if any) for this GPU
            own_pid = stressor_pids.get(gpu_id)

            # Foreign = any PID that isn't the stressor we spawned
            foreign_pids = [pid for pid in pids if pid != own_pid]

            if len(foreign_pids) == 0:
                # GPU is free or only running our stressor
                if gpu_id not in processes or not processes[gpu_id].is_alive():
                    print(f"[GPU {gpu_id}] Free. Launching stressor.")
                    proc = mp.Process(target=stress_gpu, args=(gpu_id,), daemon=True)
                    proc.start()
                    processes[gpu_id] = proc
                    stressor_pids[gpu_id] = proc.pid
                else:
                    # Avoid log spam; keep the message but you can remove it if too chatty
                    print(f"[GPU {gpu_id}] Stressor already running.")
            else:
                # GPU is busy with foreign processes
                if gpu_id in processes:
                    print(f"[GPU {gpu_id}] In use by other processes. Killing stressor.")
                    processes[gpu_id].terminate()
                    processes[gpu_id].join(timeout=5)
                    processes.pop(gpu_id, None)
                    stressor_pids.pop(gpu_id, None)
                else:
                    print(f"[GPU {gpu_id}] In use. No stressor running.")

        time.sleep(poll_interval_s)


# --------------------------------------------
# Graceful Shutdown
# --------------------------------------------
def cleanup(signum, frame) -> None:
    """
    Terminate all child processes on SIGINT/SIGTERM.

    We avoid os._exit to give children time to release CUDA contexts.
    """
    print("\n[Shutdown] Terminating all stressors.")
    for proc in mp.active_children():
        proc.terminate()
    for proc in mp.active_children():
        proc.join(timeout=5)
    # Use normal exit to allow atexit/flush
    raise SystemExit(0)


# --------------------------------------------
# Entry Point
# --------------------------------------------
if __name__ == "__main__":
    # "spawn" is the safest start method across platforms for CUDA child processes.
    mp.set_start_method("spawn", force=True)

    # Wire up clean shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Start supervising
    supervisor_loop()
