#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/split_channel_layer.hpp"
namespace caffe {

	template <typename Dtype>
	__global__ void split_data_forward_gpu_kernel(const int num, const Dtype* const bottom, Dtype * const top,
		const int channel_out, const int channel, const int inner_shape) {
		CUDA_KERNEL_LOOP(i, num) {
			int ps = i % inner_shape;
			int pc = (i / inner_shape) % channel_out;
			int pn = i / inner_shape / channel_out;
			int pidx = (pn*channel + pc)*inner_shape + ps;
			top[i] = bottom[pidx];
		}
	}
	template <typename Dtype>
	__global__ void split_data_backward_gpu_kernel(const int num, Dtype* const bottom, const Dtype * const top,
		const int channel_out, const int channel, const int inner_shape) {
		CUDA_KERNEL_LOOP(i, num) {
			int ps = i % inner_shape;
			int pc = (i / inner_shape) % channel_out;
			int pn = i / inner_shape / channel_out;
			int pidx = (pn*channel + pc)*inner_shape + ps;
			bottom[pidx] = top[i];
		}
	}
	template <typename Dtype>
	void SplitChannelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * const top_data = top[0]->mutable_gpu_data();
		const Dtype * const bottom_data = bottom[0]->gpu_data();
		int count = top[0]->count();
		split_data_forward_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, bottom_data, top_data, channel_out_, bottom[0]->channels(), bottom[0]->count(2));
		CUDA_POST_KERNEL_CHECK;
	}
	
	template <typename Dtype>
	void SplitChannelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype * const bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype * const top_diff = top[0]->gpu_diff();
		int count = top[0]->count();
		caffe_gpu_set(count, Dtype(0), bottom_diff);
		split_data_backward_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, bottom_diff, top_diff, channel_out_, bottom[0]->channels(), bottom[0]->count(2));
		
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SplitChannelLayer);

}  // namespace caffe
