#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gdn_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void GDNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		GDNParameter rm = this->layer_param_.gdn_param();
		Dtype beta_min, gamma_init, reparam_offset;
		inverse_ = rm.inverse();
		beta_min = rm.beta_min();
		gamma_init = rm.gamma_init();
		reparam_offset = rm.reparam_offset();
		h_ = bottom[0]->height();
		w_ = bottom[0]->width();
		ch_ = bottom[0]->channels();
		n_ = bottom[0]->num();
		this->blobs_.resize(2);
		this->blobs_[0].reset(new Blob<Dtype>(ch_, ch_, 1, 1));
		this->blobs_[1].reset(new Blob<Dtype>(ch_, 1, 1, 1));
		//this->blobs_[2].reset(new Blob<Dtype>(n_, ch_, h_, w_));
		pedestal = reparam_offset*reparam_offset;
		beta_bound = sqrt(beta_min + pedestal);
		gamma_bound = reparam_offset;
		Dtype * beta = this->blobs_[1]->mutable_cpu_data();
		Dtype * gamma = this->blobs_[0]->mutable_cpu_data();
		for (int i = 0; i < ch_; i++) {
			beta[i] = sqrt(Dtype(1.0) + pedestal);
			for (int j = 0; j < ch_; j++) {
				gamma[i*ch_ + j] = sqrt(pedestal);
			}
			gamma[i*ch_ + i] = sqrt(gamma_init + pedestal);
		}


		workspaceSizeInBytes = 0;
		workspaceData = NULL;
		workspace = new void*[3];
		fwd_algo_ = (cudnnConvolutionFwdAlgo_t)0;
		bwd_filter_algo_ = (cudnnConvolutionBwdFilterAlgo_t)0;
		bwd_data_algo_ = (cudnnConvolutionBwdDataAlgo_t)0;
		workspace_fwd_sizes_ = 0;
		workspace_bwd_data_sizes_ = 0;
		workspace_bwd_filter_sizes_ = 0;
		for (int g = 0; g < 3; g++) {
			CUDA_CHECK(cudaStreamCreate(&stream_[g]));
			CUDNN_CHECK(cudnnCreate(&handle_[g]));
			CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
			workspace[g] = NULL;
		}
		cudnn::createFilterDesc<Dtype>(&filter_desc_, ch_, ch_, 1, 1);
		cudnn::createTensor4dDesc<Dtype>(&bottom_descs_);
		cudnn::createTensor4dDesc<Dtype>(&top_descs_);
		cudnn::createConvolutionDesc<Dtype>(&conv_descs_);
		cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
	}
	template <typename Dtype>
	void GDNLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		h_ = bottom[0]->height();
		w_ = bottom[0]->width();
		ch_ = bottom[0]->channels();
		n_ = bottom[0]->num();
		pool_.Reshape(n_, ch_, h_, w_);
		sdata_.Reshape(n_,ch_,h_,w_);
		top[0]->Reshape(n_, ch_, h_, w_);

		size_t workspace_limit_bytes = 8 * 1024 * 1024;

		cudnn::setTensor4dDesc<Dtype>(&bottom_descs_, n_, ch_, h_, w_);
		cudnn::setTensor4dDesc<Dtype>(&top_descs_, n_, ch_, h_, w_);
		cudnn::setTensor4dDesc<Dtype>(&bias_desc_, 1, ch_, 1, 1);
		cudnn::setConvolutionDesc<Dtype>(&conv_descs_, bottom_descs_, filter_desc_, 0, 0, 1, 1);

		CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
			bottom_descs_, filter_desc_, conv_descs_, top_descs_,
			CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,	workspace_limit_bytes, &fwd_algo_));
		CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],	bottom_descs_,
			filter_desc_, conv_descs_, top_descs_,	fwd_algo_, 	&(workspace_fwd_sizes_)));

		CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_[0],
			bottom_descs_, top_descs_, conv_descs_, filter_desc_,
			CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
			workspace_limit_bytes, &bwd_filter_algo_));

		CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[0],
			bottom_descs_, top_descs_, conv_descs_, filter_desc_,
			bwd_filter_algo_, &workspace_bwd_filter_sizes_));

		CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[0],
			filter_desc_, top_descs_, conv_descs_, bottom_descs_,
			CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
			workspace_limit_bytes, &bwd_data_algo_));

		CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
			filter_desc_, top_descs_, conv_descs_, bottom_descs_,
			bwd_data_algo_, &workspace_bwd_data_sizes_));

		
		if (workspaceSizeInBytes < workspace_bwd_data_sizes_ + workspace_bwd_filter_sizes_ + workspace_fwd_sizes_) {
			workspaceSizeInBytes = workspace_bwd_data_sizes_ + workspace_bwd_filter_sizes_ + workspace_fwd_sizes_;
			cudaFree(this->workspaceData);
			cudaMalloc(&(workspaceData), workspaceSizeInBytes);
			workspace[0] = reinterpret_cast<char *>(workspaceData);
			workspace[1] = reinterpret_cast<char *>(workspaceData) + workspace_fwd_sizes_;
			workspace[2] = reinterpret_cast<char *>(workspaceData) + workspace_bwd_filter_sizes_ + workspace_fwd_sizes_;
		}
	}
	template <typename Dtype>
	void GDNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


	}
	template <typename Dtype>
	void GDNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
		
	}
	
	template <typename Dtype>
	GDNLayer<Dtype>::~GDNLayer() {

		cudnnDestroyTensorDescriptor(bottom_descs_);
		cudnnDestroyTensorDescriptor(top_descs_);
		cudnnDestroyConvolutionDescriptor(conv_descs_);
	    for (int i = 0; i < 3; i++) {
	    	cudaStreamDestroy(stream_[i]);
	    	cudnnDestroy(handle_[i]);
	    }
		cudnnDestroyFilterDescriptor(filter_desc_);
		cudnnDestroyTensorDescriptor(bias_desc_);
		cudaFree(this->workspaceData);

	}

#ifdef CPU_ONLY
	STUB_GPU(GDNLayer);
#endif

	INSTANTIATE_CLASS(GDNLayer);
	REGISTER_LAYER_CLASS(GDN);

}  // namespace caffe
