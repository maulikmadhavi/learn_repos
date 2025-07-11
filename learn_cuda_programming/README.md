# CUDA Programming Examples

This repository contains a collection of fundamental CUDA programming examples, designed to illustrate core concepts from basic memory management to more advanced parallel patterns. Each example is a self-contained `.cu` file that can be compiled and run individually.

This project was created with the assistance of the `Gemini CLI`. Thanks for providing a generous amount of context window and providing a great agentic workflow for creating this repository.

## Table of Contents

1.  [About the Examples](#about-the-examples)
    -   [Vector Addition](#1-vector-addition-vector_addcu)
    -   [Matrix Addition](#2-matrix-addition-matrix_addcu)
    -   [Matrix Multiplication](#3-matrix-multiplication-matrix_mulcu)
    -   [Softmax](#4-softmax-softmaxcu)
2.  [How to Compile and Run](#how-to-compile-and-run)
3.  [In-Depth Notes](#in-depth-notes)

---

## About the Examples

### 1. Vector Addition (`vector_add.cu`)

-   **Description:** The "Hello, World!" of CUDA. Adds two vectors (`C = A + B`).
-   **Concepts Demonstrated:**
    -   Basic host-to-device and device-to-host memory transfers (`cudaMemcpy`).
    -   Device memory allocation (`cudaMalloc`) and deallocation (`cudaFree`).
    -   Launching a simple 1D kernel.
    -   Calculating a unique global thread index in a 1D grid.

### 2. Matrix Addition (`matrix_add.cu`)

-   **Description:** Adds two 2D matrices (`C = A + B`).
-   **Concepts Demonstrated:**
    -   Launching a 2D kernel using a `dim3` grid and block structure.
    -   Mapping 2D thread and block indices to a flattened 2D matrix in memory.

### 3. Matrix Multiplication (`matrix_mul.cu`)

-   **Description:** A naive implementation of matrix multiplication (`C = A @ B`).
-   **Concepts Demonstrated:**
    -   Each thread computes a single element of the output matrix.
    -   Performing a dot product within each thread by iterating over the input matrices.
    -   A more complex 2D kernel launch.

### 4. Softmax (`softmax.cu`)

-   **Description:** A more advanced kernel that performs a row-wise softmax operation. This is a common building block in neural networks.
-   **Concepts Demonstrated:**
    -   **Shared Memory:** Using fast on-chip shared memory (`__shared__`) to facilitate cooperation between threads in a block.
    -   **Parallel Reduction:** A fundamental parallel programming pattern. The kernel performs two reductions: one to find the maximum value of a row, and another to find the sum of the exponentiated values.
    -   Dynamic shared memory allocation during kernel launch.
    -   `__syncthreads()` to ensure proper synchronization between threads in a block.

---

## How to Compile and Run

Each example can be compiled using the NVIDIA CUDA Compiler (`nvcc`).

#### 1. Compilation

Open your terminal and use the following command format. You may need to replace the path to `nvcc` with the one appropriate for your system.

```bash
# General command
/path/to/your/nvcc /path/to/example.cu -o /path/to/executable_name

# Example for matrix_add.cu
/home/maulik/.conda/envs/llm/bin/nvcc matrix_add.cu -o matrix_add_executable
```

#### 2. Execution

After successful compilation, run the executable from your terminal:

```bash
# General command
./executable_name

# Example for matrix_add_executable
./matrix_add_executable
```
The program will print a success or failure message to the console.

---

## In-Depth Notes

For a detailed explanation of the concepts used in these examples, including the host-device model, kernel launch configurations, performance comparisons, and memory profiling, please see the `cuda_programming_notes.md` file included in this repository.
