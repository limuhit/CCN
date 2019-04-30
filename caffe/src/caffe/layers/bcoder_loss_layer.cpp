#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/bcoder_loss_layer.hpp"
namespace caffe {
	template <typename Dtype>
	void BcoderLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		weight_ = (bottom.size()==3);
	}
	template <typename Dtype>
	void BcoderLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//LossLayer<Dtype>::Reshape(bottom, top);
		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
		//CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
		//	<< "Inputs must have the same dimension.";
		n_ = bottom[0]->count();
		diff_.ReshapeLike(*bottom[0]);
		inv_label_.ReshapeLike(*bottom[0]);
		tmp_.ReshapeLike(*bottom[0]);
	}
	template <typename Dtype>
	void BcoderLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype * bottom_data = bottom[0]->cpu_data();
		const Dtype * label = bottom[1]->cpu_data();
		caffe_set(n_, Dtype(1.0), inv_label_.mutable_cpu_data());
		caffe_sub(n_,  inv_label_.cpu_data(),label,inv_label_.mutable_cpu_data());
		const Dtype * inv_label = inv_label_.cpu_data();
		Dtype * tmp = tmp_.mutable_cpu_data();
		Dtype * diff = diff_.mutable_cpu_data();
		//ln(1-y)
		caffe_set(n_, Dtype(1.0), tmp);
		caffe_cpu_axpby(n_, Dtype(-1.0), bottom_data, Dtype(1.0), tmp);
		//avoid ln(0)
		caffe_add_scalar(n_, Dtype(0.000001), tmp);
		caffe_log(n_, tmp, tmp);
		//(1-t)ln(1-y)
		caffe_mul(n_,tmp,inv_label,diff);
		//(1-t)ln(1-t)
		caffe_copy(n_, inv_label, tmp);
		caffe_add_scalar(n_, Dtype(0.000001), tmp);
		caffe_log(n_, tmp, tmp);
		caffe_mul(n_, inv_label, tmp, tmp);
		//(1-t)ln(1-t)-(1-t)ln(1-y)
		caffe_cpu_axpby(n_, Dtype(1.0), tmp, Dtype(-1.0), diff);

		// tln(t)
		caffe_copy(n_, label, tmp);
		caffe_add_scalar(n_, Dtype(0.000001), tmp);
		caffe_log(n_, tmp, tmp);
		caffe_mul(n_, label, tmp, tmp);
		caffe_cpu_axpby(n_, Dtype(1.0), tmp, Dtype(1.0), diff);
		//-tln(y)
		caffe_copy(n_, bottom_data, tmp);
		caffe_add_scalar(n_, Dtype(0.000001), tmp);
		caffe_log(n_, tmp, tmp);
		caffe_mul(n_, label, tmp, tmp);
		caffe_cpu_axpby(n_, Dtype(-1.0), tmp, Dtype(1.0), diff);
		if (weight_)
			caffe_mul(n_, diff, bottom[2]->cpu_data(), diff);
		Dtype loss=caffe_cpu_asum(n_, diff);
		if (weight_)
			swt_ = caffe_cpu_asum(n_, bottom[2]->cpu_data());
		else
			swt_ = n_;
		top[0]->mutable_cpu_data()[0] = loss / swt_;
	}

	template <typename Dtype>
	void BcoderLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* label = bottom[1]->cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		Dtype* tmp = tmp_.mutable_cpu_data();
		Dtype alpha = top[0]->cpu_diff()[0] / swt_;
		Dtype * diff = diff_.mutable_cpu_data();
		const Dtype * inv_label = inv_label_.cpu_data();
		caffe_copy(n_, bottom_data, tmp);
		caffe_add_scalar(n_, Dtype(0.000001), tmp);
		caffe_div(n_, label, tmp, diff);
		caffe_cpu_axpby(n_, -alpha, diff, Dtype(0), bottom_diff);
		caffe_set(n_, Dtype(1.000001), tmp);
		caffe_sub(n_, tmp, bottom_data, tmp);
		caffe_div(n_, inv_label, tmp, diff);
		caffe_cpu_axpby(n_, alpha, diff, Dtype(1.0), bottom_diff);
		if(weight_)
			caffe_mul(n_, bottom_diff, bottom[2]->cpu_data(), bottom_diff);
	}

#ifdef CPU_ONLY
	STUB_GPU(BcoderLossLayer);
#endif

	INSTANTIATE_CLASS(BcoderLossLayer);
	REGISTER_LAYER_CLASS(BcoderLoss);

}  // namespace caffe
