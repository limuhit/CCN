#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/lot_loss_layer.hpp"
namespace caffe {
	template <typename Dtype>
	void LOTLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		LotParameter lp = this->layer_param_.lot_param();
		lone_ = (lp.method() == LotParameter::L1);
		psnr_ = (lp.method() == LotParameter::PSNR);
		slope_ = lp.slope();
		weight_=(bottom.size()==2);
	}
	template <typename Dtype>
	void LOTLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
		n_ = bottom[0]->count();
		diff_.ReshapeLike(*bottom[0]);
	}
	template <typename Dtype>
	void lot_loss_layer_sign(const int num, Dtype * const data, Dtype slope) {
		for (int i = 0; i < num; i++)
		{
			if (data[i] < 0)
				data[i] = -slope;
			else if (data[i] == 0)
				data[i] = 0;
			else
				data[i] = 1;
		}
	}
	template <typename Dtype>
	void LOTLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype * bottom_data = bottom[0]->cpu_data();
		Dtype * diff = diff_.mutable_cpu_data();
		caffe_copy(n_, bottom_data, diff);
		Dtype loss;
		if (lone_)
			lot_loss_layer_sign<Dtype>(n_, diff, slope_);
		if (weight_)
			caffe_mul<Dtype>(n_, diff, bottom[1]->cpu_data(), diff);
		loss = caffe_cpu_dot(n_, diff, bottom_data);
		if (weight_){
			Dtype base_ = caffe_cpu_asum(n_, bottom[1]->cpu_data());
			if (base_ == 0)
				base_+=1;
			top[0]->mutable_cpu_data()[0] = loss / base_;
		}
		else {
			top[0]->mutable_cpu_data()[0] = loss / n_;
		}
		if (psnr_) {
			mse_ = top[0]->cpu_data()[0];
			top[0]->mutable_cpu_data()[0] = -log10(2550 * 255.0 / mse_);
		}
			 
		
	}

	template <typename Dtype>
	void LOTLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		Dtype * diff = diff_.mutable_cpu_data();
		Dtype alpha = 1.0 / bottom[0]->count()*top[0]->cpu_diff()[0];
		if (psnr_) alpha = alpha / mse_;
		if (weight_)
			alpha = 1.0 / base_;
		caffe_cpu_axpby(n_, alpha, diff, Dtype(0), bottom_diff);
	}

#ifdef CPU_ONLY
	STUB_GPU(LOTLossLayer);
#endif

	INSTANTIATE_CLASS(LOTLossLayer);
	REGISTER_LAYER_CLASS(LOTLoss);

}  // namespace caffe
