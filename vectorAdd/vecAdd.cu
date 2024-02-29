
/*****************************************************************************
 * File:        vector_add_unified_memory.cu
 * Description: Parallel Vector Addition with Unified Memory. A + B = C
 *              
 * Compile:     nvcc -O3 -o vecAdd vecAdd.cu
 * Run:         ./vecAdd
 *                  
 * 1. Allocate Unified Memory
 * 2. Copy data to Unified Memory
 * 3. Launch the kernel
 * 4. Copy the result back to the host
 * 5. Free the Unified Memory
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vecAddKernel(float* A,float* B, float* C, int numElements);

int main(int argc, char* argv[]){

    // numElements is the size of the vector
    int numElements = 1048576;
    // size is the size of the vector in bytes
    size_t size = numElements * sizeof(float);
    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < numElements; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }


    // Allocate vectors in device memory
    float* d_A;
    float* d_B;
    float* d_C;
    // cudaMalloc returns a pointer to the allocated memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory 
    printf("Copy input data from the host memory to the CUDA device\n");
    // cudaMemcpy is a synchronous operation, and returns once the data has been copied
    //cudaMemcpyHostToDevice is a flag that specifies the direction of the copy
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Allocate CUDA events for estimating
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the Vector Add CUDA Kernel
    // threadsPerBlock is the number of threads in each thread block
    int threadsPerBlock = 256;
    // blocksPerGrid is the number of thread blocks in the grid
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    // vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaEventRecord(start);
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaEventRecord(stop);

    // Copy result from device memory to host memory
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


 // Verify that the result vector is correct (sampling)
    printf("Verifying vector addition...\n");
    for (int idx = 0; idx < numElements; idx++) {
        printf("[INDEX %d] %f + %f = %f\n", idx, h_A[idx], h_B[idx], h_C[idx]);
        if (fabs(h_A[idx] + h_B[idx] - h_C[idx]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d\n", idx);
            exit(EXIT_FAILURE);
        }
    }
    printf(".....\n");
    printf("Test PASSED\n");
    // Compute and Print the performance
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    double flopsPerVecAdd = static_cast<double>(numElements);
    double gigaFlops = (flopsPerVecAdd * 1.0e-9f) / (msecTotal / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size = %.0f Ops, "
           "WorkgroupSize= %u threads/block\n",
           gigaFlops, msecTotal, flopsPerVecAdd, threadsPerBlock);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

}

__global__
void vecAddKernel(float* A, float* B, float* C, int numElements){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements)
        C[i] = A[i] + B[i];
}



