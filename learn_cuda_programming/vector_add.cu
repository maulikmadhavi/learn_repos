#include <iostream>
#include <vector>

// CUDA Kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Host function to print a vector
void printVector(const std::vector<float>& vec) {
    for (float f : vec) {
        std::cout << f << " ";
    }
    std::cout << std::endl;
}

int main() {
    // 1. Set up host data
    int n = 1000;
    std::vector<float> h_A(n);
    std::vector<float> h_B(n);
    std::vector<float> h_C(n);

    // Initialize host vectors
    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // 2. Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size = n * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 3. Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // 4. Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // 5. Copy results from device to host
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // 6. Verify the results
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (h_C[i] != (h_A[i] + h_B[i])) {
            std::cout << "Error at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Vector addition successful!" << std::endl;
        // std::cout << "Result: ";
        // printVector(h_C); // Optional: print the full vector
    }


    // 7. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
