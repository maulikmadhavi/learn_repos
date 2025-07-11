#include <iostream>
#include <vector>
#include <cstdlib>

// Naive matrix multiplication kernel
__global__ void matrixMul(const float *A, const float *B, float *C, int M, int N, int K) {
    // Each thread computes one element of C
    // C has dimensions M x N
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        // Dot product of row from A and column from B
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // Matrix dimensions
    // A: M x K
    // B: K x N
    // C: M x N
    int M = 128;
    int K = 256;
    int N = 64;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 1. Host memory allocation and initialization
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // 2. Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 3. Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    // 4. Launch the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // 5. Copy result from device to host
    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    // 6. Verify the results on the host (CPU-based calculation)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += h_A[row * K + i] * h_B[i * N + col];
            }
            h_C_ref[row * N + col] = sum;
        }
    }

    bool success = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(h_C[i] - h_C_ref[i]) > 1e-3) { // Larger tolerance for float precision
            std::cout << "Error at index " << i << ": GPU=" << h_C[i] << ", CPU=" << h_C_ref[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Matrix multiplication successful!" << std::endl;
    } else {
        std::cout << "Matrix multiplication failed!" << std::endl;
    }

    // 7. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
