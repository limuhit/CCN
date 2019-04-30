#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/float2int_layer.hpp"
#include "curand.h"
#include <curand_kernel.h>
namespace caffe {
	template <typename Dtype>
	__global__ void float2int_gpu_kernel(const int num, const Dtype* const bottom, Dtype * const top, 
		const float * noise,bool train) {
		CUDA_KERNEL_LOOP(i, num) {
			Dtype ta = bottom[i];
			if (train) {
				top[i] = ta + static_cast<Dtype>(noise[i]) - 0.5;
			}
			else {
				top[i] = static_cast<Dtype>(roundf(ta));
			}
		}
	}
	template <typename Dtype>
	void Float2IntLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* const top_data = top[0]->mutable_gpu_data();
		float * noise = noise_.mutable_gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		int count = bottom[0]->count();
		//LOG(INFO) << "gpu_int";
		int seed = time(NULL);
		if (train_) {
			curandGenerator_t g_randgen;
			curandCreateGenerator(&g_randgen, CURAND_RNG_PSEUDO_DEFAULT);
			curandSetPseudoRandomGeneratorSeed(g_randgen, seed);
			curandGenerateUniform(g_randgen, noise, count);
		}
		float2int_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, bottom_data, top_data, noise,train_);
		CUDA_POST_KERNEL_CHECK;
	}
	
	template <typename Dtype>
	void Float2IntLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		caffe_gpu_memcpy(top[0]->count() * sizeof(Dtype), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
		
	}

	INSTANTIATE_LAYER_GPU_FUNCS(Float2IntLayer);

}  // namespace caffe
