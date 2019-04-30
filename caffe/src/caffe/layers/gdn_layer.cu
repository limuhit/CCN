#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gdn_layer.hpp"
__global__ void sync_gdn_groups() { }
namespace caffe {
	template <typename Dtype>
	void GDNLayer<Dtype>::gpu_conv_forward(const Dtype * bottom_data, Dtype * top_data) {

		const Dtype * weight = this->blobs_[0]->gpu_data();
		const Dtype* bias_data = this->blobs_[1]->gpu_data();

		CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
			cudnn::dataType<Dtype>::one,
			bottom_descs_, bottom_data,	filter_desc_, weight, conv_descs_,
			fwd_algo_, workspace[0], workspace_fwd_sizes_,
			cudnn::dataType<Dtype>::zero, top_descs_, top_data ));
		CUDNN_CHECK(cudnnAddTensor(handle_[0],
			cudnn::dataType<Dtype>::one,
			bias_desc_, bias_data ,
			cudnn::dataType<Dtype>::one,
			top_descs_, top_data));
		sync_gdn_groups << <1, 1 >> >();
	}
	template <typename Dtype>
	void GDNLayer<Dtype>::gpu_conv_backward(Dtype * bottom_diff, const Dtype * top_diff, const Dtype * bottom_data) {

		const Dtype * weight = this->blobs_[0]->gpu_data();
		Dtype * weight_diff = this->blobs_[0]->mutable_gpu_diff();
		Dtype* bias_diff  = this->blobs_[1]->mutable_gpu_diff();
		// Gradient w.r.t. bias.

		CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0],
			cudnn::dataType<Dtype>::one,
			top_descs_, top_diff , cudnn::dataType<Dtype>::one,
			bias_desc_, bias_diff));

		// Gradient w.r.t. weights.
		CUDNN_CHECK(cudnnConvolutionBackwardFilter(
			handle_[1],	cudnn::dataType<Dtype>::one,
			bottom_descs_, bottom_data,	top_descs_, top_diff,
			conv_descs_, bwd_filter_algo_, workspace[1],
			workspace_bwd_filter_sizes_, cudnn::dataType<Dtype>::one,
			filter_desc_, weight_diff));
			
		CUDNN_CHECK(cudnnConvolutionBackwardData(
			handle_[2],	cudnn::dataType<Dtype>::one, filter_desc_, weight,
			top_descs_, top_diff ,	conv_descs_,
			bwd_data_algo_, workspace[2], workspace_bwd_data_sizes_,
			cudnn::dataType<Dtype>::zero, bottom_descs_, bottom_diff ));
				
		sync_gdn_groups << <1, 1 >> >();
	}
	template<typename Dtype>
	__global__ void gdn_forward_check_weight_gpu_kernel(const int count, Dtype * data, const Dtype threshold) {
		CUDA_KERNEL_LOOP(i, count) {
			if (data[i] < threshold)
				data[i] = threshold;
		}
	}
	template <typename Dtype>
	void GDNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* const top_data = top[0]->mutable_gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype * pool = pool_.mutable_gpu_data();
		Dtype * sdata = sdata_.mutable_gpu_data();
		int cnt = bottom[0]->count();
		gdn_forward_check_weight_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(ch_), CAFFE_CUDA_NUM_THREADS >> >
			(ch_, this->blobs_[1]->mutable_gpu_data(), beta_bound);
		gdn_forward_check_weight_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(ch_*ch_), CAFFE_CUDA_NUM_THREADS >> >
			(ch_*ch_, this->blobs_[0]->mutable_gpu_data(), gamma_bound);
		caffe_gpu_sqr(cnt, bottom_data, sdata);
		//caffe_gpu_memcpy(cnt * sizeof(Dtype), sdata, this->blobs_[2]->mutable_gpu_data());
		gpu_conv_forward(sdata,pool);
		caffe_gpu_sqrt(cnt, pool, sdata);
		if(inverse_)
			caffe_gpu_mul(cnt, bottom_data, sdata, top_data);
		else
			caffe_gpu_div(cnt, bottom_data, sdata, top_data);

		CUDA_POST_KERNEL_CHECK;
	}
	template<typename Dtype>
	__global__ void gdn_backward_check_gradient_gpu_kernel(const int count, Dtype * grad, const Dtype * data,
		const Dtype threshold) {
		CUDA_KERNEL_LOOP(i, count) {
			if (data[i] < threshold && grad[i] >= 0)
				grad[i] = 0;
		}
	}
	template <typename Dtype>
	void GDNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* const top_diff = top[0]->gpu_diff();
		Dtype* const bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* top_data = top[0]->gpu_data();
		Dtype * pool = pool_.mutable_gpu_data();
		const Dtype * sdata = sdata_.mutable_gpu_data();
		Dtype * sdiff = sdata_.mutable_gpu_diff();
		Dtype * pdiff = pool_.mutable_gpu_diff();
		int cnt = bottom[0]->count();
		if (inverse_) {
			caffe_gpu_mul(cnt, top_diff, sdata, bottom_diff);
			caffe_gpu_mul(cnt, top_diff, bottom_data, sdiff);
			caffe_gpu_axpby(cnt, Dtype(0), top_diff, Dtype(0.5), sdiff);
		}
		else {
			caffe_gpu_div(cnt, top_diff, sdata, bottom_diff);
			caffe_gpu_mul(cnt, bottom_diff, top_data, sdiff);
			caffe_gpu_axpby(cnt, Dtype(0), top_diff, Dtype(-0.5), sdiff);
		}
		caffe_gpu_div(cnt, sdiff, sdata, sdiff);
		caffe_gpu_sqr(cnt, bottom_data, sdata_.mutable_gpu_data());
		gpu_conv_backward(pdiff,sdiff,sdata);
		caffe_gpu_mul(cnt, pdiff, bottom_data, pdiff);
		caffe_gpu_axpby(cnt, Dtype(2.0), pdiff, Dtype(1.0), bottom_diff);
		gdn_backward_check_gradient_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(ch_), CAFFE_CUDA_NUM_THREADS >> >
			(ch_,this->blobs_[1]->mutable_gpu_diff(),this->blobs_[1]->gpu_data(),beta_bound);
		gdn_backward_check_gradient_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(ch_*ch_), CAFFE_CUDA_NUM_THREADS >> >
			(ch_*ch_, this->blobs_[0]->mutable_gpu_diff(), this->blobs_[0]->gpu_data(), gamma_bound);
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(GDNLayer);

}  // namespace caffe
