#ifndef _CAFFE_CUDA_MACRO
#define _CAFFE_CUDA_MACRO

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32
#define FULL_MASK 0xffffffff
// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 1024;

inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK \
do { \
  cudaError_t err = cudaGetLastError(); \
  if (cudaSuccess != err) \
      printf("CUDA kernel failed : %s\n", cudaGetErrorString(err)); \
} while (0)

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error!=cudaSuccess){printf("%s\n",cudaGetErrorString(error));} \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    if(status!=CUBLAS_STATUS_SUCCESS){printf("%s\n",cublasGetErrorString(status));} \
  } while (0)
#endif