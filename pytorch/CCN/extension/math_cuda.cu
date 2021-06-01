#include "math_functions.hpp"
//#include "include/ext_all.hpp" 
#include <curand.h>

void caffe_gpu_dot(cublasHandle_t handle, const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(handle, n, x, 1, y, 1, out));
}
void caffe_gpu_dot(cublasHandle_t handle, const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(handle, n, x, 1, y, 1, out));
}

void caffe_gpu_gemv(cublasHandle_t handle, const CBLAS_TRANSPOSE TransA, const int M,
  const int N, const float alpha, const float* A, const float* x,
  const float beta, float* y) {
cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
CUBLAS_CHECK(cublasSgemv(handle, cuTransA, N, M, &alpha,
    A, N, x, 1, &beta, y, 1));
}


void caffe_gpu_gemv(cublasHandle_t handle, const CBLAS_TRANSPOSE TransA, const int M,
  const int N, const double alpha, const double* A, const double* x,
  const double beta, double* y) {
cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
CUBLAS_CHECK(cublasDgemv(handle, cuTransA, N, M, &alpha,
    A, N, x, 1, &beta, y, 1));
}


template <typename scalar_t>
__global__ void set_kernel(const int n, const scalar_t alpha,scalar_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}


void caffe_gpu_set(cudaStream_t stream, const int N, const double alpha, double* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(double) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, alpha, Y);
}

void caffe_gpu_set(cudaStream_t stream, const int N, const float alpha, float* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(float) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, alpha, Y);
}

void caffe_gpu_set(cudaStream_t stream, const int N, const int alpha, int* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(int) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<int><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, alpha, Y);
}


void caffe_gpu_scal(cublasHandle_t handle, const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(handle, N, &alpha, X, 1));
}

void caffe_gpu_scal(cublasHandle_t handle, const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(handle, N, &alpha, X, 1));
}
void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

cublasHandle_t getHandle(cudaStream_t stream){
    cublasHandle_t handle;
	cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return 0;
    }
    stat = cublasSetStream(handle, stream);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return 0;
    }
    return handle;
}

const char* cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "Unknown cublas status";
  }

  int sphere_cal_npart_hw(const int height, const int width, const int npart, float* weight, int * tidx, int * hinv){
    int heights_per_part = height / npart;
    int lefts = height % npart;
    int lefta = lefts / 2;
    int leftb = lefts - lefta;
    for(int i=0;i<npart;i++){
        tidx[i] = heights_per_part;
    }
    if(npart%2==0){
        tidx[npart/2-1] += lefta;
        tidx[npart/2] += leftb;
    }else{
        tidx[npart/2] += lefts;
    }
    for(int i=0;i<npart-1;i++){
        tidx[i+1] += tidx[i];
    }
    float pi = acos(-1.0);
    if(npart%2==0){
        for(int i =0;i<npart/2-1;i++)
            tidx[i+npart] = static_cast<int>(weight[i]*width * cos(((tidx[i] - 0.5)/height - 0.5)*pi)+0.5);
        tidx[npart/2-1+npart] = width;
        tidx[npart/2 + npart] = width;
        for(int i = npart/2+1;i<npart;i++)
            tidx[i+npart] = static_cast<int>(weight[i]*width * cos(((tidx[i-1] + 0.5)/height - 0.5)*pi)+0.5);
    }else{
        for(int i =0;i<npart/2;i++)
            tidx[i+npart] = static_cast<int>(weight[i]*width * cos(((tidx[i] - 0.5)/height - 0.5)*pi)+0.5);
        tidx[npart/2 + npart] = width;
        for(int i = npart/2+1;i<npart;i++)
            tidx[i+npart] = static_cast<int>(weight[i]*width * cos(((tidx[i-1] + 0.5)/height - 0.5)*pi)+0.5);
    }
    
    for(int i=0, j=0; i<npart; i++){
      for(int k=j; k < tidx[i]; k++){
        //printf("%d %d %d\n",k,i,k-j);
        hinv[k] = i;
        hinv[k+height] = k-j;
      }
      j = tidx[i];
    }
    //for(int i=0;i<height*2;i++) printf("%d ", hinv[i]);
    //printf("\n");

    return tidx[npart/2]-tidx[npart/2-1];
  }