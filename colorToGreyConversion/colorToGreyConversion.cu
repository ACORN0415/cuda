/*****************************************************************************
 * File:        convertColorToGrey.cu
 * Description: Convert color scale to grey scale of input image.
 *              This program doesn't save result image, and just show the result.
 *              For reading image, OpenCV library should be used.
 *              
 * Compile:     nvcc -o convertColorToGrey convertColorToGrey.cu -I.. -lcuda $(pkg-config opencv4 --libs --cflags)
 * Run:         ./convertColorToGrey <image file path>
 *****************************************************************************/

//opencv is not ready so if i download opencv, i will do next time

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Define the number of threads in a block
#define CHANNELS 3


void Usage(char* prog_name);
//parameter : input image, output image, width, height
// why use unsigned char* instead of Mat?
// because Mat is a class of OpenCV, and it is not supported in CUDA.
// so why use unsigned char? because it is the same as uchar in OpenCV.
__global__ void convertColorToGrey(unsigned char* input, unsigned char* output, int width, int height);

int main(int argc, char** argv)
{

    if (argc != 2) Usage(argv[0]);
    //
    const char* file_name = argv[1];
    int width, height, chnnels;
    unsigned char*h_origImg, *h_resultImg;
    cv::Mat origImg = cv:::imread(file_name);

    //
    width = origImg.cols;
    height = origImg.rows;
    channels = origImg.channels();
    printf("Image size = (%d, %d, x %d)\n", width, height, channels);
    //allocate memory for original image and result image
    assert(channels == CHANNELS);
    //cv::Mat half meas half size of original image
    cv::Mat half;
    //cv resize function : resize the image, how? -> cv::Size() -> width, height
    cv::resize(origImg, half, cv::Size(), width/2, height/2);
    //
    cv::imshow("Original Image", half);
    cv::waitKey(0);
    //
    h_origImg = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));
    h_resultImg = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    (void)memcpy(h_origImg, origImg.data, width * height * channels * sizeof(unsigned char));

    //allocate memory for device
    unsigned char* d_origImg, *d_resultImg;
    cudaMalloc((void**)&d_origImg, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_resultImg, width * height * sizeof(unsigned char));

    cudaMemcpy(d_origImg, h_origImg, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    const int block_size = 16;
    dim3 threads(block_size, block_size);
    //ceil means round up-> 반올림 cieil -> 올림, floor -> 버림 , round -> 반올림
    dim3 grid(ceil(width / (double)threads.x), ceil(height / (double)threads.y));
    convertColorToGrey<<<grid, threads>>>(d_origImg, d_resultImg, width, height);

    cudaMemcpy(h_resultImg, d_resultImg, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cv::Mat resultImg(height, width, CV_8UC1);
    memcpy(resultImg.data, h_resultImg, width * height);

    cudaFree(d_origImg);
    cudaFree(d_resultImg);

    free(h_origImg);
    free(h_resultImg);

    cv::resize(resultImg, resultImg, cv::Size(), width/2, height/2);
    cv::Imshow("Result Image", resultImg);
    cv::waitKey(0);

    return 0;

}

void Usage(char* prog_name)
{
    fprintf(stderr, "usage: %s <image file path>\n", prog_name);
    exit(0);
}


// convert color scale to grey scale
__global__ void convertColorToGrey(unsigned char* input, unsigned char* output, int width, int height)
{
    //coloumn = x, row = y
    //전체에서 idx 를 구하는 방법
    //threadIdx는 block 내에서의 thread의 index
    //blockIdx는 grid 내에서의 block의 index
    //blockDim은 block 내에서의 thread의 개수
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    //if the pixel is in the image
    if(row<height && col<width)
    {
        //offset = row * width + col
        int offset = row * width + col;
        //offset * 3 because of RGB
        int rgbOffset = offset * CHANNELS;
       
        unsigned char r = input[rgbOffset];
        unsigned char g = input[rgbOffset + 1];
        unsigned char b = input[rgbOffset + 2];
        //
        output[offset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}