#ifndef CAFFE_MSSIM_LOSS_LAYER_HPP_
#define CAFFE_MSSIM_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/util/im2col.hpp"
namespace caffe {

	template <typename Dtype>
	class MssimLossLayer : public LossLayer<Dtype> {
	public:
		explicit MssimLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}
		~MssimLossLayer();
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "MssimLoss"; }
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}
	protected:
		bool handles_setup_;
		cudnnHandle_t handle_[5];
		cudaStream_t  stream_[5];

		// algorithms for forward and backwards convolutions

		cudnnConvolutionFwdAlgo_t fwd_algo_[5];
		cudnnConvolutionBwdDataAlgo_t bwd_data_algo_[5];
		cudnnTensorDescriptor_t bottom_descs_[5], top_descs_[5];
		cudnnFilterDescriptor_t      filter_desc_;
		cudnnConvolutionDescriptor_t conv_descs_[5];
		size_t workspace_fwd_sizes_[5];
		size_t workspace_bwd_data_sizes_[5];
		size_t workspaceSizeInBytes;  // size of underlying storage
		void *workspaceData;  // underlying storage
		void **workspace;

		void cuda_conv_forward(Dtype ** bottom, Dtype ** top);
		void cuda_conv_backward(Dtype ** bottom, Dtype ** top);
		void cuda_reshape();
		void cuda_setup();
		void multi_sqr(Dtype ** x, Dtype **y);
		void multi_mul(Dtype **x,  Dtype **y, Dtype **z);
		void multi_axpby(Dtype **x, Dtype **y, Dtype alpha);
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		Blob<Dtype> x_[5],x2_[5],xy_[5],y_[5],y2_[5],tmp_[5];
		Blob<Dtype> la_[5],lcs_[5],sim_[5];
		Blob<Dtype> data_[5], label_[5];
		Blob<Dtype> ones_;
		Blob<Dtype> diff_;
		Blob<Dtype> weight_;
		
		int out_h_[5], out_w_[5];
		Dtype mcs_[5];
		int n_,ch_,h_,w_;
		Dtype c1_, c2_;
		const Dtype gamma_[5] = { Dtype(0.0448), Dtype(0.2856), Dtype(0.3001),Dtype(0.2363), Dtype(0.1333) };
	
	};
}

#endif  