#ifndef CAFFE_GDN_LAYER_HPP_
#define CAFFE_GDN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class GDNLayer : public Layer<Dtype> {
	public:
		explicit GDNLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "GDN"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
		~GDNLayer();
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		void gpu_conv_forward(const Dtype * bottom_data, Dtype * top_data);
		void gpu_conv_backward(Dtype * bottom_diff, const Dtype * top_diff, const Dtype * bottom_data);
		cudnnHandle_t handle_[3];
		cudaStream_t  stream_[3];
		cudnnConvolutionFwdAlgo_t fwd_algo_;
		cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
		cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
		cudnnTensorDescriptor_t bottom_descs_, top_descs_;
		cudnnTensorDescriptor_t    bias_desc_;
		cudnnFilterDescriptor_t      filter_desc_;
		cudnnConvolutionDescriptor_t conv_descs_;
		size_t workspace_fwd_sizes_;
		size_t workspace_bwd_data_sizes_;
		size_t workspace_bwd_filter_sizes_;
		size_t workspaceSizeInBytes;  // size of underlying storage
		void *workspaceData;  // underlying storage
		void **workspace;
		int n_, ch_, w_, h_; 
		
		Dtype pedestal, beta_bound , gamma_bound;
		Blob<Dtype> pool_,sdata_;
		bool inverse_;
	};
}

#endif  