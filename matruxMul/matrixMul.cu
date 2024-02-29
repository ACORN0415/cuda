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
void matrixMul(float* A, float* B, float* C, int M, int K, int N){
    //blockIdx를 정의 하지 않아도 되는 이유는 blockIdx는 그리드의 인덱스를 나타내는 변수이기 때문
    //
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDIm.x + threadIdx.x;
    //M*N matrix
    if(row<M && col<N){
        float value = 0;
        for(int i=0; i<K;i++){
            //행렬 곱 A행렬 * B행렬
            //row행 col열
            //row*K+i : A행렬의 row행 i열
            //i*N+col : B행렬의 i행 col열
            // A행렬의 row행 i열 * B행렬의 i행 col열
            // 행렬곱인데 idx부분이 단일 숫자인 이유는? 일차원 벡터로 표현했기 떄문
            // 일차원 벡터로 어떻게 표현했는가? 
            // A행렬의 row행 i열은 A[row*K+i]로 표현 이유는? 
            // B행렬의 i행 col열은 B[i*N+col]로 표현 이유는?
            // A행렬 row행 i열을 왜 A[row*K+i]로 표현했는가? 
            // row행 1열row행 2열 row행 3열 ... row행 K열
            // B행렬 i행 col열을 왜 B[i*N+col]로 표현했는가?
            // 1행 col열 2행 col열 3행 col열 ... K행 col열
            //i*N+col인 이유는 col열이 N개이기 때문         
            value += A[row*K+i]*B[i*N+col];
        }
        C[row*N+col] = value;
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