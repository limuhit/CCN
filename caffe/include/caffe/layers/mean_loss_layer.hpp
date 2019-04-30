#ifndef CAFFE_MEAN_LOSS_LAYER_HPP_
#define CAFFE_MEAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
namespace caffe {

	template <typename Dtype>
	class MeanLossLayer : public LossLayer<Dtype> {
	public:
		explicit MeanLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "MeanLoss"; }
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactTopBlobs() const { return 1; }
		void set_mean(Dtype *m, int nclass = 1);
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		Blob<Dtype> mean_;
		Blob<Dtype> tmp_;
		Blob<Dtype> diff_;
		Blob<Dtype> ones_;
		int h_, w_, ch_, num_;

	};
}

#endif  
