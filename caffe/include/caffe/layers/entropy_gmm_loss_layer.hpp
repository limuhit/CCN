#ifndef CAFFE_ENTROPY_GMM_LOSS_LAYER_HPP_
#define CAFFE_ENTROPY_GMM_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
namespace caffe {

	template <typename Dtype>
	class EntropyGmmLossLayer : public LossLayer<Dtype> {
	public:
		explicit EntropyGmmLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline int ExactNumBottomBlobs() const { return 4; }
		virtual inline int ExactTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "EntropyGmmLoss"; }
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}
		//void set_weight(const Dtype *weight);
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		//Blob<Dtype> count;
		Blob<Dtype> diff_;
		int n_, c_,h_,w_;
		int label_dim_, num_;

	};
}

#endif  