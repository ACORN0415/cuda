#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

template<typename T>
inline void common_random_init_vector(T *vec, int n)
{
    for (int i = 0; i < n; i++) {
        vec[i] = rand() / (T)RAND_MAX;
    }
}

template<typename T>
inline void common_random_init_matrix(T *mat, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i*n + j] = rand() / (T)RAND_MAX;
        }
    }
}