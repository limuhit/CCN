#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mtranspose_layer.hpp"
namespace caffe {
	template <typename Dtype>
	__global__ void mtranspose_forward_gpu_kernel(const int nthreads, const Dtype * const bottom, Dtype * const top,
		const int inner_size, const int channel, const int groups) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int pn = index / inner_size / channel;
			int pc = (index / inner_size) % channel;
			int ps = index % inner_size;
			int tidx = (pn*inner_size*channel / groups + pc / groups*inner_size + ps)*groups + pc%groups;
			top[tidx] = bottom[index];
		}
	}
	template <typename Dtype>
	__global__ void mtranspose_forward_gpu_kernel_inverse(const int nthreads, const Dtype * const bottom, Dtype * const top,
		const int inner_size, const int channel, const int groups) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int pn = index / inner_size / channel;
			int pc = (index / inner_size) % channel;
			int ps = index % inner_size;
			int tidx = (pn*inner_size*channel / groups + pc / groups*inner_size + ps)*groups + pc%groups;
			//top[tidx] = bottom[index];
			top[index] = bottom[tidx];
		}
	}
	template <typename Dtype>
	void MTransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		int num = bottom[0]->count();
		if (inverse_) {
			mtranspose_forward_gpu_kernel_inverse<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
				(num, bottom_data, top[0]->mutable_gpu_data(), inner_size_, channel_, groups_);
		}
		else {
			mtranspose_forward_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
				(num, bottom_data, top[0]->mutable_gpu_data(), inner_size_, channel_, groups_);
		}
		
		CUDA_POST_KERNEL_CHECK;
	}
	template <typename Dtype>
	__global__ void mtranspose_backward_gpu_kernel(const int nthreads, Dtype * const bottom, const Dtype * const top,
		const int inner_size, const int channel, const int groups) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int pn = index / inner_size / channel;
			int pc = (index / inner_size) % channel;
			int ps = index % inner_size;
			int tidx = (pn*inner_size*channel / groups + pc / groups*inner_size + ps)*groups + pc%groups;
			bottom[index] = top[tidx];
		}
	}
	template <typename Dtype>
	void MTransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int num = bottom[0]->count();
		if (!inverse_) {
			mtranspose_backward_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
				(num, bottom[0]->mutable_gpu_diff(), top[0]->gpu_diff(), inner_size_, channel_, groups_);
			CUDA_POST_KERNEL_CHECK;
		}
		
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MTransposeLayer);

}  // namespace caffe
