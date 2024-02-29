#include <stdio.h>


__global__ void hello(void);
int main() {

    hello<<<1,1>>>();
    printf("Hello, World!\n");
    return 0;
}

__global__ void hello(void){
    printf("Hello, Cuda!\n");
}