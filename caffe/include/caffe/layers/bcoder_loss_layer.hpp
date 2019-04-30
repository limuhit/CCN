#ifndef CAFFE_BCODER_LOSS_LAYER_HPP_
#define CAFFE_BCODER_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
namespace caffe {

	template <typename Dtype>
	class BcoderLossLayer : public LossLayer<Dtype> {
	public:
		explicit BcoderLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline int MinBottomBlobs() const { return 2; }
		virtual inline int ExactNumBottomBlobs() const { return -1; }
		virtual inline int ExactTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "BcoderLoss"; }
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
		Blob<Dtype> inv_label_;
		Blob<Dtype> diff_;
		Blob<Dtype> tmp_;
		bool weight_;
		Dtype swt_;
		int n_;
	};
}

#endif  