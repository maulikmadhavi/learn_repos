# CUDA Programming Notes: From Concepts to Performance

This document summarizes the key concepts of CUDA programming, connecting high-level frameworks like PyTorch to the low-level operations happening on the GPU.

## 1. Introduction to CUDA
- **CUDA (Compute Unified Device Architecture):** An NVIDIA platform for general-purpose computing on GPUs (GPGPU). It allows you to use the massive parallel processing power of a GPU for tasks beyond graphics.
- **Core Idea:** Offload computationally intensive parts of your program to the GPU, while the main application logic runs on the CPU.

## 2. Core CUDA Concepts

### The Host-Device Model
The fundamental concept in CUDA is the separation between the CPU and the GPU.
- **Host:** The CPU and its memory (your computer's RAM).
- **Device:** The GPU and its memory (VRAM).

A typical CUDA workflow involves these steps:
1.  **Allocate Memory on Device:** Reserve space in the GPU's VRAM for your data (`cudaMalloc`).
2.  **Copy Data to Device:** Move input data from Host RAM to Device VRAM (`cudaMemcpy`).
3.  **Launch Kernel:** Execute the parallel computation on the Device (`myKernel<<<...>>>`).
4.  **Copy Data to Host:** Move the results from Device VRAM back to Host RAM (`cudaMemcpy`).

### The Execution Hierarchy: Grid, Blocks, and Threads
When you launch a kernel (`<<<...>>>`), you define its execution configuration. This is a three-level hierarchy:
- **Threads:** The fundamental unit of execution. One thread runs one copy of your kernel code.
- **Blocks:** A group of threads. Threads within a block can cooperate using fast shared memory and synchronization.
- **Grid:** A group of blocks. A single kernel launch creates one grid.

The configuration `<<<blocksPerGrid, threadsPerBlock>>>` tells the GPU how to structure its thousands of cores for your specific problem. Each thread has built-in variables (`threadIdx`, `blockIdx`, `blockDim`) to calculate its unique global index and determine which piece of data to work on.

## 3. CUDA Programming Examples

### Example 1: Vector Addition
This is the "Hello, World!" of CUDA. The kernel is simple: one thread is responsible for adding one pair of elements.
- **Kernel Logic:** `C[i] = A[i] + B[i];`
- **Grid Configuration:** A 1D grid of 1D blocks is typically used.

### Example 2: Self-Attention in a Transformer
A complex operation like self-attention is a sequence of simpler, chained CUDA kernel launches.
**Formula:** `Attention(Q, K, V) = softmax( (Q @ Kᵀ) / √dₖ ) @ V`

| Self-Attention Step | PyTorch Code | Low-Level CUDA Operation |
| :--- | :--- | :--- |
| 1. **Create Q, K, V** | `X @ W_Q` | 3x Matrix-Matrix Multiply |
| 2. **Get Scores** | `Q @ K.T` | 1x Matrix-Matrix Multiply |
| 3. **Scale** | `scores / scale` | 1x Element-wise Kernel |
| 4. **Softmax** | `softmax(scores)` | 1x Complex Fused Kernel |
| 5. **Apply to V** | `weights @ V` | 1x Matrix-Matrix Multiply |

Each line in the PyTorch code triggers at least one highly optimized CUDA kernel from libraries like **cuBLAS** and **cuDNN**.

## 4. PyTorch vs. Raw CUDA: A Performance Comparison

| Aspect | PyTorch | Raw CUDA C++ | Winner |
| :--- | :--- | :--- | :--- |
| **Standard Ops** (MatMul, Conv) | Uses hyper-optimized NVIDIA libraries (cuBLAS, cuDNN). | A self-written implementation will be naive by comparison. | **PyTorch** (by a large margin) |
| **Novel/Complex Ops** | Chains many simple kernels, incurring launch and memory overhead. | Can **fuse** many operations into a single kernel, eliminating overhead. | **Raw CUDA** (if well-written) |
| **Development Speed** | Extremely fast. Write one line of Python. | Very slow. Requires writing, compiling, and debugging C++/CUDA. | **PyTorch** |

**Key Takeaway:** Stick with PyTorch 99% of the time. Only write a custom CUDA kernel when you have profiled your model and found a significant bottleneck in a non-standard, chainable operation.

## 5. Understanding GPU Memory Usage

GPU memory usage is more complex than just the model's weights. The main consumers are:
1.  **Model Parameters:** The weights and biases of your network.
2.  **Gradients:** Stored for every parameter during the backward pass (doubles parameter memory).
3.  **Optimizer States:** Can be a huge consumer. Adam, for example, stores two moving averages for each parameter (triples or quadruples parameter memory).
4.  **Activations:** The output of each layer, saved during the forward pass for use in the backward pass. **Directly proportional to batch size.**
5.  **Workspace Memory:** Temporary memory ("scratchpad") required by optimized kernels (like cuDNN convolutions) to run efficiently.
6.  **PyTorch Caching Allocator:** PyTorch reserves a large pool of memory to avoid slow calls to the OS/driver. This is why `nvidia-smi` often shows high memory usage even when the model isn't running.

### Why Memory Usage Varies

You may notice that memory usage for the same model (e.g., ResNet-18) changes across different PyTorch versions or different GPU hardware. This is not a bug; it's a feature of a highly optimized system.

#### Reason 1: Software Changes (Different PyTorch Versions)
- **cuDNN Version:** Newer PyTorch versions link against newer cuDNN libraries. A new cuDNN might select a different, faster algorithm for a convolution that requires more workspace memory.
- **Allocator Strategy:** The internal logic of PyTorch's caching memory allocator is constantly being improved, which can change the total reserved memory.
- **New Features:** Features like `torch.compile` can fuse operations, leading to **lower** memory usage by avoiding intermediate storage.

#### Reason 2: Hardware Differences (Different GPUs like A100 vs. H100)
- **Hardware-Aware Kernels:** cuDNN queries the specific GPU it's running on and selects algorithms optimized for that hardware's unique features (e.g., Tensor Cores, FP8 support).
- **Tensor Cores:** Different GPU generations have different Tensor Cores (e.g., Ampere vs. Hopper). Algorithms designed for Hopper's FP8 support on an H100 are unavailable on an A100 and have a completely different memory profile.
- **Memory Hierarchy:** Different GPUs have different cache sizes and memory bandwidth. The optimal algorithm for an A100's memory subsystem may be different from the one for an L4 or H100.

**Conclusion:** The variation in memory usage is a direct result of the PyTorch/cuDNN software stack intelligently adapting to the underlying hardware to give you the best possible **performance**. The fastest algorithm is often not the most memory-frugal one.


