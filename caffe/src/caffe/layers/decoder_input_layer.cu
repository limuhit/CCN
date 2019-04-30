#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/decoder_input_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void decoder_input_gpu_kernel(const int num, const Dtype * const input,
		const int * index, Dtype * const output, const int start_idx, const int len_idx,
		const int height, const int width, const int ngroup, const int nchannel, const int psum) {
		CUDA_KERNEL_LOOP(i, num) {
			int tg = i % ngroup;
			int tl = (i / ngroup) % len_idx;
			int tn = i / ngroup / len_idx;
			int thw = index[tl + start_idx];
			int tw = thw % width;
			int th = thw / width;
			int tc = psum - tw - th;
			int pidx = ((tn*nchannel + tc)*ngroup + tg)*height*width + thw;
			output[pidx] = input[i];
		}

	}
	template <typename Dtype>
	void DecoderInputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * top_data = top[0]->mutable_gpu_data();
		const Dtype * bottom_data = bottom[0]->gpu_data();
		const int* index = index_.gpu_data();
		int psum = pidx_;
		pidx_ = (pidx_ + 1) % mod_;
		if (psum == 0) {
			caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
		}
		else {
			psum -= 1;
			//int st = psum > height_ + width_ - 2 ? psum - height_ - width_ + 2 : 0;
			//int end = psum > height_ + width_ - 2 ? height_ + width_ - 1 : psum + 1;
			int st = psum - nchannel_ + 1 < 0 ? 0 : psum - nchannel_ + 1;
			int end = psum < height_ + width_ - 2 ? psum + 1 : height_ + width_ - 1;
			int len_idx = start_idx_[end] - start_idx_[st];
			int count = len_idx*num_ * 1;
			decoder_input_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, bottom_data, index, top_data, start_idx_[st], len_idx, height_, width_, 1, nchannel_, psum);
		}
		
		
	}
	template <typename Dtype>
	void DecoderInputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

	INSTANTIATE_LAYER_GPU_FUNCS(DecoderInputLayer);

}  // namespace caffe
