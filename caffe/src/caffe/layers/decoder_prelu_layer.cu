#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/decoder_prelu_layer.hpp"

namespace caffe {
	template<typename Dtype>
	__global__ void decoder_prelu_forward_gpu_kernel(const int num, const Dtype * const input, const Dtype * const slope,
		Dtype * const output, const int h_start, const int h_len, const int channel, const int height,
		const int width, const int plan_sum) {
		CUDA_KERNEL_LOOP ( i , num ) {
			int ph = h_start + i % h_len;
			int pw = plan_sum - ph;
			int pc = (i / h_len) % channel;
			int pn = i / h_len / channel;
			int idx = ((pn*channel + pc)*height + ph)*width + pw;
			if (input[idx] < 0)
				output[idx] = slope[pc] * input[idx];
			else
				output[idx] = input[idx];
		}
	}
	template<typename Dtype>
	__global__ void decoder_prelu_copy_forward_gpu_kernel(const int num, const Dtype * const input,
		Dtype * const output, const int h_start, const int h_len, const int channel, const int height,
		const int width, const int plan_sum) {
		CUDA_KERNEL_LOOP(i, num) {
			int ph = h_start + i % h_len;
			int pw = plan_sum - ph;
			int pc = (i / h_len) % channel;
			int pn = i / h_len / channel;
			int idx = ((pn*channel + pc)*height + ph)*width + pw;
			output[idx] = input[idx];
		}
	}
	template <typename Dtype>
	void DecoderPreluLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* slope_data = this->blobs_[0]->gpu_data();
		int plan_sum = index_;
		index_ = (index_ + 1) % mod_;
		int h_st = (plan_sum >= w_ ? plan_sum - w_ + 1 : 0);
		int h_end = (plan_sum >= h_ ? h_ : plan_sum + 1);
		int h_len = h_end - h_st;
		if (bottom[0] == top[0]) {
			decoder_prelu_copy_forward_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(h_len*ch_*num_), CAFFE_CUDA_NUM_THREADS >> >
				(h_len*ch_*num_, bottom_data, bottom_memory_.mutable_cpu_data(), h_st, h_len, ch_, h_, w_, plan_sum);
		}
		decoder_prelu_forward_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(h_len*ch_*num_), CAFFE_CUDA_NUM_THREADS >> >
			(h_len*ch_*num_, bottom_data, slope_data, top_data, h_st, h_len, ch_, h_, w_, plan_sum);
	}
	template<typename Dtype>
	__global__ void decoder_prelu_backward_gpu_kernel(const int num, const Dtype * const top_diff, const Dtype * const slope,
		const Dtype * const bottom_data, Dtype * const bottom_diff, const int inner_shape, const int channel) {
		CUDA_KERNEL_LOOP(i, num) {
			int c = (i / inner_shape) % channel;
			bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
				+ slope[c] * (bottom_data[i] <= 0));
		}
	}
	template <typename Dtype>
	void DecoderPreluLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* slope_data = this->blobs_[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		const int count = bottom[0]->count();
		if (top[0] == bottom[0])
			bottom_data = bottom_memory_.gpu_data();
		decoder_prelu_backward_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, top_diff, slope_data, bottom_data,	bottom[0]->mutable_gpu_diff(), h_*w_, ch_);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DecoderPreluLayer);

}  // namespace caffe
