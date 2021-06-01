#pragma once
#include "ext_all.hpp"
#include "math.h"

enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, AtlasConj=114};

const char* cublasGetErrorString(cublasStatus_t error);

void caffe_gpu_gemv(cublasHandle_t handle, const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x, const float beta, float* y);
void caffe_gpu_gemv(cublasHandle_t handle, const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x, const double beta, double* y);

void caffe_gpu_dot(cublasHandle_t handle, const int n, const float* x, const float* y,  float* out);
void caffe_gpu_dot(cublasHandle_t handle, const int n, const double* x, const double* y, double * out);

void caffe_gpu_set(cudaStream_t stream, const int N, const float alpha, float* Y);
void caffe_gpu_set(cudaStream_t stream, const int N, const double alpha, double* Y);
void caffe_gpu_set(cudaStream_t stream, const int N, const int alpha, int* Y);

void caffe_gpu_scal(cublasHandle_t handle, const int N, const float alpha, float *X);

void caffe_gpu_scal(cublasHandle_t handle, const int N, const double alpha, double *X);
void caffe_gpu_memcpy(const size_t N, const void* X, void* Y);

int sphere_cal_npart_hw(const int height, const int width, const int npart, float* weight, int * tidx, int * hinv);

cublasHandle_t getHandle(cudaStream_t stream);