#include <iostream>
#include <vector>
#include <cstdlib> // For rand()

// Kernel for 2D matrix addition
__global__ void matrixAdd(const float *A, const float *B, float *C, int rows, int cols) {
    // Calculate the unique 2D thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to avoid accessing out-of-bounds memory
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

void printMatrix(const std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Matrix dimensions
    int rows = 256;
    int cols = 256;
    int numElements = rows * cols;
    size_t size = numElements * sizeof(float);

    // 1. Host memory allocation and initialization
    std::vector<float> h_A(numElements);
    std::vector<float> h_B(numElements);
    std::vector<float> h_C(numElements);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 2. Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 3. Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // 4. Launch the kernel
    // Define 2D block and grid dimensions
    dim3 threadsPerBlock(16, 16); // 16*16 = 256 threads per block
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    // 5. Copy result from device to host
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // 6. Verify the results on the host
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (std::abs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            std::cout << "Error at index " << i << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Matrix addition successful!" << std::endl;
    } else {
        std::cout << "Matrix addition failed!" << std::endl;
    }

    // 7. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
