#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // For std::max_element
#include <numeric>   // For std::accumulate

// A block-level softmax kernel. Each block handles one row.
// This demonstrates a common parallel reduction pattern using shared memory.
__global__ void softmax(const float *in, float *out, int rows, int cols) {
    // Each block processes one row of the input matrix.
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Allocate shared memory for the current row.
    // The size should be the number of columns.
    extern __shared__ float s_row[];

    // 1. Find the maximum value in the row (Parallel Reduction)
    // ---------------------------------------------------------
    // Each thread loads one element from global to shared memory
    if (tid < cols) {
        s_row[tid] = in[row * cols + tid];
    }
    __syncthreads();

    // Iteratively find the max value in the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_row[tid] = fmaxf(s_row[tid], s_row[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = s_row[0];
    __syncthreads();

    // 2. Compute exponentials and their sum (Parallel Reduction)
    // ---------------------------------------------------------
    // Each thread computes exp(x - max) and stores it back to shared memory
    if (tid < cols) {
        s_row[tid] = expf(in[row * cols + tid] - max_val);
    }
    __syncthreads();

    // Iteratively sum the exponentials
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_row[tid] += s_row[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = s_row[0];
    __syncthreads();

    // 3. Divide by the sum to get the final softmax value
    // ---------------------------------------------------------
    if (tid < cols) {
        // **FIX:** Reload the original value, re-calculate the exponential, and then divide.
        // The value in s_row[tid] was overwritten by the summation reduction.
        out[row * cols + tid] = expf(in[row * cols + tid] - max_val) / sum_exp;
    }
}

int main() {
    // Matrix dimensions
    int rows = 4;
    int cols = 256; // Must be <= threadsPerBlock for this simple kernel
    int numElements = rows * cols;
    size_t size = numElements * sizeof(float);

    // 1. Host memory allocation and initialization
    std::vector<float> h_in(numElements);
    std::vector<float> h_out(numElements);
    std::vector<float> h_out_ref(numElements);

    for (int i = 0; i < numElements; ++i) {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX * 5.0f;
    }

    // 2. Device memory allocation
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // 3. Copy data from host to device
    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);

    // 4. Launch the kernel
    // We launch one block per row.
    // The number of threads per block must be >= number of columns for this naive implementation.
    int threadsPerBlock = 256; // Must be a power of 2 for the reduction to work easily
    int blocksPerGrid = rows;
    
    // We need to allocate shared memory dynamically
    size_t sharedMemSize = cols * sizeof(float);

    softmax<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_out, rows, cols);

    // 5. Copy result from device to host
    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);

    // 6. Verify the results on the host
    for (int i = 0; i < rows; ++i) {
        const float* row_start = h_in.data() + i * cols;
        float max_val = *std::max_element(row_start, row_start + cols);
        
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum_exp += std::exp(row_start[j] - max_val);
        }

        for (int j = 0; j < cols; ++j) {
            h_out_ref[i * cols + j] = std::exp(row_start[j] - max_val) / sum_exp;
        }
    }

    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (std::abs(h_out[i] - h_out_ref[i]) > 1e-5) {
            std::cout << "Error at index " << i << ": GPU=" << h_out[i] << ", CPU=" << h_out_ref[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Softmax successful!" << std::endl;
    } else {
        std::cout << "Softmax failed!" << std::endl;
    }

    // 7. Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
