#ifndef CAFFE_LOT_LOSS_LAYER_HPP_
#define CAFFE_LOT_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
namespace caffe {

	template <typename Dtype>
	class LOTLossLayer : public LossLayer<Dtype> {
	public:
		explicit LOTLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int ExactNumBottomBlobs() const{ return -1; }
		virtual inline int ExactTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "LOTLoss"; }
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}
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
		bool lone_;
		int n_;
		Dtype base_;
		Dtype slope_;
		bool weight_;
		bool psnr_;
		Dtype mse_;
	};
}

#endif  