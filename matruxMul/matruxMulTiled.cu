/*****************************************************************************
 * File:        matrixMul.cu
 * Description: Matrix multiplication, C = AB
 *              A has m x k dimensions, B has k x n dimensions, and C has
 *              m x n dimensions.
 *              It is not for the most performance.
 *              
 * Compile:     nvcc -o matrixMul matrixMul.cu -I.. -lcuda
 * Run:         ./matrixMul <m> <k> <n>
 *                  <m> : the number of rows in Matrix A
 *                  <k> : the number of columns in Matrix A, it is also
 *                        the number of rows in Matrix B.
 *                  <n> : the number of columns in Matrix B.
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void Usage(char* prog_name);
void common_random_init_matrix(float* matrix, int row, int col);
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int K, int N);

int main(int argc, char* argv[]){
    if(argc !=r){
        Usage(argv[0]);
    }
    int M = strtol(argv[1], NULL, 10);
    int K = strtol(argv[2], NULL, 10);
    int N = strtol(argv[3], NULL, 10);
    printf("[Matrix multiplication, C=AB]\n");
    printf("Matrix A(%d x %d) * Matrix B(%d x %d) = Matrix C(%d x %d)\n", M, K, K, N, M, N);

    float* h_A = (float*)malloc(sizeof(float)*M*K);
    float* h_B = (float*)malloc(sizeof(float)*K*N);
    float* h_C = (float*)malloc(sizeof(float)*M*N);

    if(h_A==NULL || h_B==NULL || h_C==NULL){
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    //initialize the host matrix
    common_random_init_matrix<float>(h_A, M, K);
    common_random_init_matrix<float>(h_B, K, N);

    //allocate device memory
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(float)*M*K);
    cudaMalloc((void**)&d_B, sizeof(float)*K*N);
    cudaMalloc((void**)&d_C, sizeof(float)*M*N);

    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float)*K*N, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int BLOCK_SIZE = 16;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);
    //dim3 dimGrid(ceil(N/thread.x), ceil(M/thread.y));
    printf("Launch the kernel\n");
    printf("dimGrid(%d, %d), dimBlock(%d, %d)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
    
    //cudaDeviceSynchronize function means that the CPU thread waits until the GPU kernel is finished
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);

    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_C, d_C, sizeof(float)*M*N, cudaMemcpyDeviceToHost);


    

}

__global__
void matrixMulTiled(float* A, float* B, float* C, int M, int K, int N){
    //blockIdx를 정의 하지 않아도 되는 이유는 blockIdx는 그리드의 인덱스를 나타내는 변수이기 때문
    __shared__ float Asub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bsub[TILE_WIDTH][TILE_WIDTH];

    //bx = block의 x좌표, by = block의 y좌표
    int bx = blockIdx.x;, by = blockIdx.y;
    //tx = thread의 x좌표, ty = thread의 y좌표
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row  = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;

    float p value

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDIm.x + threadIdx.x;
    float pvalue = 0;
    for (int ph = 0; ph<ceil(K/(float)TILE_WIDTH); ++ph){
        if(row<M && ph*TILE_WIDTH+tx<K){
            Asub[ty][tx] = A[row*K+ph*TILE_WIDTH+tx];
        }else{
            Asub[ty][tx] = 0;
        }
        if(ph*TILE_WIDTH+ty<N && col<N){
            Bsub[ty][tx] = B[(ph*TILE_WIDTH+ty)*N+col];
        }else{
            Bsub[ty][tx] = 0;
        }
        __syncthreads();
        for(int k=0; k<TILE_WIDTH; k++){
            pvalue += Asub[ty][k]*Bsub[k][tx];
        }
        __syncthreads();
    }
    if(row<M && col<N){
        C[row*N+col] = pvalue;
    }
    
}

void Usage(char prog_name[])
{
    fprintf(stderr, "Usage: %s <m> <k> <n>\n", prog_name);
    fprintf(stderr, "\t<m> : the number of rows in matrix A.\n");
    fprintf(stderr, "\t<k> : the number of columns in Matrix A, it is also\n");
    fprintf(stderr, "\t      the number of rows in Matrix B.\n");
    fprintf(stderr, "\t<n> : the number of columns in matrix B.\n");
    exit(EXIT_FAILURE);
}

void common_random_init_matrix(float* matrix, int row, int col){
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            matrix[i*col+j] = (float)rand()/(float)RAND_MAX;
        }
    }
}

int verify_matrix_mul(float* A, float* B, float* C, int M, int K, int N){
    float* temp = (float*)malloc(sizeof(float)*M*N);
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            float value = 0;
            for(int k=0; k<K; k++){
                value += A[i*K+k]*B[k*N+j];
            }
            temp[i*N+j] = value;
        }
    }
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            if(fabs(temp[i*N+j]-C[i*N+j])>1e-5){
                printf("Error: C[%d][%d] = %f, temp[%d][%d] = %f\n", i, j, C[i*N+j], i, j, temp[i*N+j]);
                return 0;
            }
        }
    }
    return 1;
}