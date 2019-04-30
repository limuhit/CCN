#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/decoder_quant_layer.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void decoder_quant_forward_kernel(const int nthreads, const Dtype* const input, const Dtype * const weight,
		Dtype * const output, const int inner_shape, const int channel, const int level) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tc = (index / inner_shape) % channel;
			int idx = static_cast<int>(input[index]);
			output[index] = weight[tc*level + idx];
		}
	}
	template <typename Dtype>
	__global__ void decoder_quant_weight_kernel(const int nthreads, const Dtype* const input, Dtype * const output, const int level) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			output[index*level] = input[index*level];
			for (int i = 1; i < level; i++) {
				output[index*level + i] = output[index*level+i-1]+exp(input[index*level + i]);
			}
		}
	}
	template <typename Dtype>
	void DecoderQuantLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * const top_data = top[0]->mutable_gpu_data();
		const Dtype * const bottom_data = bottom[0]->gpu_data();
		if (!init_) {
			decoder_quant_weight_kernel<Dtype> << <CAFFE_GET_BLOCKS(ch_), CAFFE_CUDA_NUM_THREADS >> >
				(ch_, this->blobs_[0]->gpu_data(),weight_.mutable_gpu_data(),levels_);
			init_ = true;
		}
		int count = bottom[0]->count();
		decoder_quant_forward_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, bottom_data, weight_.gpu_data(), top_data, h_*w_, ch_, levels_);
		
	}
	
	template <typename Dtype>
	void DecoderQuantLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DecoderQuantLayer);

}  // namespace caffe
