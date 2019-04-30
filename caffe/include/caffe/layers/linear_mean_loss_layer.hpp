#ifndef CAFFE_LINEAR_MEAN_LOSS_LAYER_HPP_
#define CAFFE_LINEAR_MEAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
namespace caffe {

	template <typename Dtype>
	class LinearMeanLossLayer : public LossLayer<Dtype> {
	public:
		explicit LinearMeanLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline int MaxNumBottomBlobs() const { return 2; }
		virtual inline int ExactTopBlobs() const { return 1; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline const char* type() const { return "LinearMeanLoss"; }
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}
		void set_direction(const Dtype d){ direction_ = d; }
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		Blob<Dtype> diff_;
		Blob<Dtype> ones_;
		int n_;
		Dtype direction_;
	};
}

#endif  