#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/lot_loss_layer.hpp"
namespace caffe {
	template <typename Dtype>
	__global__ void lot_loss_layer_sign_kernel(const int num, Dtype * const data, Dtype slope) {
		CUDA_KERNEL_LOOP(i,num)
		{
			if (data[i] < 0)
				data[i] = -slope;
			else if (data[i] == 0)
				data[i] = 0;
			else
				data[i] = 1;
		}
	}
	template <typename Dtype>
	void LOTLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype * bottom_data = bottom[0]->gpu_data();
		Dtype * diff = diff_.mutable_gpu_data();
		caffe_gpu_memcpy(n_*sizeof(Dtype), bottom_data, diff);
		Dtype loss;
		if (lone_)
			lot_loss_layer_sign_kernel<Dtype> << <CAFFE_GET_BLOCKS(n_), CAFFE_CUDA_NUM_THREADS >> >(n_, diff, slope_);
		if (weight_)
			caffe_gpu_mul<Dtype>(n_, diff, bottom[1]->gpu_data(), diff);
		caffe_gpu_dot(n_, diff, bottom_data,&loss);
		if (weight_) {
			caffe_gpu_asum(n_, bottom[1]->cpu_data(),&base_);
			if (base_ == 0)
				base_ += 1;
			top[0]->mutable_cpu_data()[0] = loss / base_;
		}
		else {
			top[0]->mutable_cpu_data()[0] = loss / n_;
		}
		if (psnr_) {
			mse_ = top[0]->cpu_data()[0];
			top[0]->mutable_cpu_data()[0] = 10*log10(255.0 * 255.0 / mse_);
		}
	}

	template <typename Dtype>
	void LOTLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		Dtype * diff = diff_.mutable_gpu_data();
		Dtype alpha = 1.0 / bottom[0]->count()*top[0]->cpu_diff()[0];
		if (psnr_) alpha = 10* alpha / mse_;
		if (weight_)
			alpha = 1.0 / base_;
		caffe_gpu_axpby(n_, alpha, diff, Dtype(0), bottom_diff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(LOTLossLayer);

}  // namespace caffe
