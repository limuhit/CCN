#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/entropy_gmm_loss_layer.hpp"
#include "math.h"

namespace caffe {
	template <typename Dtype>
	void EntropyGmmLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
	
	}
	template <typename Dtype>
	void EntropyGmmLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		n_ = bottom[0]->num();
		w_ = bottom[0]->width();
		h_ = bottom[0]->height();
		c_ = bottom[0]->channels();
		int cnt = bottom[0]->count();
		num_ = bottom[3]->count();
		label_dim_ = cnt / num_;
		CHECK_EQ(cnt, bottom[1]->count()) << "The size of the two bottom blobs should be same.";
		CHECK_EQ(cnt, bottom[2]->count()) << "The size of the two bottom blobs should be same.";
		CHECK_EQ(cnt, num_*label_dim_) << "The size of the parameter should be the multipler of the label.";
		diff_.Reshape(n_, c_, h_, w_);
	}
	
	template <typename Dtype>
	void EntropyGmmLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		
	
	}

	template <typename Dtype>
	void EntropyGmmLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
		
	}

#ifdef CPU_ONLY
	STUB_GPU(EntropyGmmLossLayer);
#endif

	INSTANTIATE_CLASS(EntropyGmmLossLayer);
	REGISTER_LAYER_CLASS(EntropyGmmLoss);

}  // namespace caffe
