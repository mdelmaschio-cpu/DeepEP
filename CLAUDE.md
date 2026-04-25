# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

DeepEP is a CUDA/C++ communication library for Mixture-of-Experts (MoE) expert parallelism. It implements high-throughput all-to-all GPU kernels for dispatching tokens to experts and combining results, targeting NVLink (intranode) and RDMA (internode) networks with optional FP8 support. The Python API wraps a PyTorch `CUDAExtension`.

## Build

```bash
# Build in-place (requires NVSHMEM for internode/low-latency features)
NVSHMEM_DIR=/path/to/nvshmem python setup.py build

# After build, symlink the compiled .so into the package directory
ln -s build/lib.*/deep_ep_cpp.*.so .

# Install as a wheel
bash install.sh
```

Key environment variables that affect what gets compiled:

| Variable | Effect |
|---|---|
| `NVSHMEM_DIR` | Path to NVSHMEM; required for internode and low-latency kernels |
| `TORCH_CUDA_ARCH_LIST` | GPU architectures (default `'9.0'` for H800; use `'8.0'` for A100) |
| `DISABLE_SM90_FEATURES` | Set to `1` for A100 builds (disables SM90+ PTX instructions) |
| `DISABLE_AGGRESSIVE_PTX_INSTRS` | Set to `1` to disable undefined-behavior PTX load/store tricks |
| `TOPK_IDX_BITS` | `32` or `64` for expert index width |

Without `NVSHMEM_DIR`, only intranode kernels (`intranode.cu`) are compiled; `internode.cu` and `internode_ll.cu` are excluded.

The `csrc/CMakeLists.txt` is for debugging only and is not the standard build path.

## Lint and Format

```bash
# Install linting tools
pip install -r requirements-lint.txt

# Format files changed relative to origin/main
bash format.sh

# Check formatting only (used in CI)
bash format.sh --check
```

Three tools are enforced: `yapf` (Python), `ruff` (Python linting), `clang-format` (C++/CUDA). CI runs on every push and PR via `.github/workflows/format.yml`. Column limit is 140 for both Python and C++.

## Tests

Tests require a configured distributed environment. Edit `tests/utils.py` `init_dist()` to match your cluster before running:

```bash
python tests/test_intranode.py    # NVLink-only, single node
python tests/test_internode.py    # Multi-node RDMA + NVLink
python tests/test_low_latency.py  # Inference low-latency kernels
```

Tests use PyTorch distributed (NCCL backend) and are driven by `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and `RANK` environment variables. `tests/utils.py` provides shared helpers: `bench()` (microbenchmark with L2 flush), `calc_diff()` (tensor comparison), FP8 cast utilities, and `bench_kineto()` for CUDA profiling.

## Architecture

### Three Kernel Families

The library exposes three distinct communication paths, each in its own `.cu` file:

1. **`csrc/kernels/intranode.cu`** — NVLink only, single node. Highest throughput (~153–158 GB/s on H800). Used for training/prefill when all ranks are on one node.

2. **`csrc/kernels/internode.cu`** — Hybrid NVLink + RDMA across nodes. Implements the asymmetric-bandwidth forwarding algorithm from DeepSeek-V3: tokens travel over NVLink within the RDMA sender's local group, then cross nodes via RDMA. Used for training/prefill with EP across multiple nodes.

3. **`csrc/kernels/internode_ll.cu`** — Low-latency pure-RDMA path. Targets inference decoding (<200 µs). Supports hook-based communication–computation overlap (SM-free recv) and double-batch overlapping, and is CUDA graph compatible. Also uses NVLink for intranode transfers within this path.

### Layer Breakdown

```
Python (deep_ep/)
  └── buffer.py        ← Buffer class: high-level dispatch/combine logic, config selection
  └── utils.py         ← EventOverlap wrapper, NVLink validation
      ↓ pybind11
C++ (csrc/)
  └── deep_ep.cpp      ← Python bindings, Buffer C++ class methods
  └── deep_ep.hpp      ← Buffer class declaration, SharedMemoryAllocator
  └── config.hpp       ← Config struct (tuning params), LowLatencyBuffer/Layout structs
  └── event.hpp        ← EventHandle (shared CUDA event wrapper)
      ↓ CUDA kernels
csrc/kernels/
  ├── runtime.cu       ← Device init and synchronization setup
  ├── layout.cu        ← Dispatch layout computation (token → rank/expert mapping)
  ├── intranode.cu     ← NVLink all-to-all kernels
  ├── internode.cu     ← Hybrid RDMA+NVLink kernels
  ├── internode_ll.cu  ← Low-latency RDMA kernels
  └── ibgda_device.cuh ← InfiniBand GPUDirect Async NVSHMEM primitives
```

### Python Buffer Class (`deep_ep/buffer.py`)

`Buffer` is the sole user-facing class. It owns the GPU communication buffers and selects the appropriate kernel family based on topology. Key design points:

- **Static SM count** (`Buffer.set_num_sms(n)`) controls how many SMs the normal kernels use; default is 20. This is a global setting.
- **Config maps** in `buffer.py` (dicts keyed by EP rank count) hold pre-tuned `Config` objects. When EP size is outside the map range, the nearest entry is used.
- `dispatch()` and `combine()` auto-route to intranode, internode, or raise if the wrong method is called. `internode_dispatch/combine()` and `low_latency_dispatch/combine()` are explicit.
- `get_dispatch_layout()` must be called before `dispatch()` to compute which tokens go where; it returns a layout tensor that is an input to dispatch.
- Low-latency mode uses a separate dual-buffer memory layout (`LowLatencyBuffer` in `config.hpp`) with mask buffers for signaling readiness.

### Memory Model

- **Intranode**: IPC-shared GPU memory accessed peer-to-peer via NVLink. `SharedMemoryAllocator` in `deep_ep.hpp` abstracts CUDA IPC vs. Fabric API allocation.
- **Internode**: NVSHMEM symmetric heap for RDMA; IPC shared buffers for local NVLink forwarding.
- **Low-latency**: Dual-buffered symmetric NVSHMEM allocation (`LowLatencyLayout`) with separate odd/even send and receive buffers for overlap.

### Key Conventions

- All C++ assertions use `EP_HOST_ASSERT` / `EP_DEVICE_ASSERT` macros from `exception.cuh`. Do not use raw `assert`.
- `topk_idx_t` (`deep_ep.topk_idx_t`) is the expert-index dtype; it resolves to `torch.int64` by default but can be `torch.int32` if compiled with `TOPK_IDX_BITS=32`.
- CUDA event-based synchronization is wrapped in `EventHandle` (C++) and `EventOverlap` (Python). Do not use raw `torch.cuda.Event` in new code—go through these wrappers to stay compatible with CUDA graph capture.
- When adding new kernel configurations, add entries to both the dispatch and combine config maps in `buffer.py`, keyed by EP rank count.
