/*****************************************************************************
 * File:        imageBlur.cu
 * Description: Blur input image using 3D blocks.
 *              This program doesn't save result image, and just show the result.
 *              For reading image, OpenCV library should be used.
 *              
 * Compile:     nvcc -o imageBlur imageBlur.cu -I.. -lcuda $(pkg-config opencv4 --libs --cflags)
 * Run:         ./imageBlur <image file path>
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define BLUR_SIZE 10
#define CHANNELS 3

void Usage(char* prog_name[]);
void blurKernel(unsigned char* input, unsigned char* output, int width, int height, int channels);

//argc means the number of arguments, and argv is the array of arguments.
int main(int argc, char* argv[]){
    int (argc !=2){Usage(argv[0]);}
    
    const char* file_name = argv[1];
    int width, height, channels;
    unsigned char* h_original_image, h_resultImg;
    unsigned char* d_original_image, d_resultImg;

    cv::Mat original_image = cv::imread(file_name);

    width = original_image.cols;
    height = original_image.rows;
    channels = original_image.channels();
    printf("Image size: %d x %d x %d\n", width, height, channels);
    assert(channels == CHANNELS);

    cv:Mat half;
    cv::resize(original_image, half, cv::Size(width/2,height/2));
    cv::imshow("Original Image", half);
    cv::waitKey(0);

    h_original_image = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));
    h_resultImg = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));
    (void)memcpy(h_original_image, original_image.data, width * height * channels);

    cudaMalloc(&d_original_image, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_resultImg, width * height * channels * sizeof(unsigned char));

    cudaMemcpy(d_original_image, h_original_image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    const int block_size = 16;
    dim3 threads = dim3(block_size, block_size, channels);
    dim3 blocks = dim3(ceil(width/double(threads.x)), ceil(height/double(threads.y)));
    blurKernel<<<blocks, threads>>>(d_original_image, d_resultImg, width, height, channels);

    cudaMemcpy(h_resultImg, d_resultImg, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat resultImg(height, width, CV_8UC3);
    (void)memcpy(resultImg.data, h_resultImg, width * height * channels);
    cudaFree(d_original_image);
    cudaFree(d_resultImg);

    free(h_original_image);
    free(h_resultImg);

    cv::resize(resultImg, half, cv::Size(width/2, height/2));
    cv::imshow("Result Image", resultImg);
    cv::waitKey(0);
    
    return 0;
    //...colorToGrey와 일치


}

__global__
void blurKernel(unsigned char* input, unsigned char* output, int width, int height, int channels){
    
    // col, row idx of the image
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int plane = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (row<height && col< width && plane<channels){
        //pixel value and count
        // pixelVal is the sum of pixel values in the block
        // pixelCnt is the number of pixels in the block
        int pixelVal = 0;
        int pixelCnt = 0;
        //bRow, bCol is the index of the block
        for(int bRow = -BLUR_SIZE; bRow<BLUR_SIZE+1; bRow++){
            for(int bCol = -BLUR_SIZE; bCol<BLUR_SIZE+1; bCol++){
                int curRow = row + bRow;
                int curCol = col + bCol;
                //check if the pixel is in the image
                if(curRow >= 0 && curRow < height && curCol >= 0 && curCol < width){
                    //add pixel value to pixelVal and increase pixelCnt
                    pixelVal += input[(curRow * width + curCol) * channels + plane];
                    pixelCnt++;
                }
            }
        }
    }
  
}