#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/bcoder_loss_layer.hpp"
namespace caffe {
	template <typename Dtype>
	void BcoderLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype * bottom_data = bottom[0]->gpu_data();
		const Dtype * label = bottom[1]->gpu_data();
		caffe_gpu_set(n_, Dtype(1.0), inv_label_.mutable_gpu_data());
		caffe_gpu_axpby(n_, Dtype(-1.0), label, Dtype(1.0), inv_label_.mutable_gpu_data());
		const Dtype * inv_label = inv_label_.gpu_data();
		Dtype * tmp = tmp_.mutable_gpu_data();
		Dtype * diff = diff_.mutable_gpu_data();
		caffe_gpu_set(n_, Dtype(1.0), tmp);
		caffe_gpu_axpby(n_, Dtype(-1.0), bottom_data, Dtype(1.0), tmp);
		//avoid ln(0)
		caffe_gpu_add_scalar(n_, Dtype(0.000001), tmp);
		caffe_gpu_log(n_, tmp, tmp);
		//(1-t)ln(1-y)
		caffe_gpu_mul(n_, tmp, inv_label, diff);
		//(1-t)ln(1-t)
		caffe_gpu_memcpy(n_*sizeof(Dtype), inv_label, tmp);
		caffe_gpu_add_scalar(n_, Dtype(0.000001), tmp);
		caffe_gpu_log(n_, tmp, tmp);
		caffe_gpu_mul(n_, inv_label, tmp, tmp);
		//(1-t)ln(1-t)-(1-t)ln(1-y)
		caffe_gpu_axpby(n_, Dtype(1.0), tmp, Dtype(-1.0), diff);

		// tln(t)
		caffe_gpu_memcpy(n_*sizeof(Dtype), label, tmp);
		caffe_gpu_add_scalar(n_, Dtype(0.000001), tmp);
		caffe_gpu_log(n_, tmp, tmp);
		caffe_gpu_mul(n_, label, tmp, tmp);
		caffe_gpu_axpby(n_, Dtype(1.0), tmp, Dtype(1.0), diff);
		//-tln(y)
		caffe_gpu_memcpy(n_*sizeof(Dtype), bottom_data, tmp);
		caffe_gpu_add_scalar(n_, Dtype(0.000001), tmp);
		caffe_gpu_log(n_, tmp, tmp);
		caffe_gpu_mul(n_, label, tmp, tmp);
		caffe_gpu_axpby(n_, Dtype(-1.0), tmp, Dtype(1.0), diff);
		Dtype loss;
		top[0]->mutable_cpu_data()[0] = loss / n_;
		if (weight_)
			caffe_gpu_mul(n_, diff, bottom[2]->gpu_data(), diff);
		caffe_gpu_asum(n_, diff, &loss);;
		if (weight_)
			caffe_gpu_asum(n_, bottom[2]->gpu_data(),&swt_);
		else
			swt_ = n_;
		top[0]->mutable_cpu_data()[0] = loss / swt_;
	}
	
	template <typename Dtype>
	void BcoderLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype* diff = diff_.mutable_gpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		Dtype* tmp = tmp_.mutable_gpu_data();
		Dtype alpha = top[0]->cpu_diff()[0] / swt_;
		const Dtype * label = bottom[1]->gpu_data();
		const Dtype * inv_label = inv_label_.gpu_data();
		caffe_gpu_memcpy(n_*sizeof(Dtype), bottom[0]->gpu_data(), tmp);
		caffe_gpu_add_scalar(n_, Dtype(0.000001), tmp);
		caffe_gpu_div(n_, label, tmp, diff);
		caffe_gpu_axpby(n_, -alpha, diff, Dtype(0), bottom_diff);
		caffe_gpu_set(n_, Dtype(1.000001), tmp);
		caffe_gpu_sub(n_, tmp, bottom[0]->gpu_data(), tmp);
		caffe_gpu_div(n_, inv_label, tmp, diff);
		caffe_gpu_axpby(n_, alpha, diff, Dtype(1.0), bottom_diff);
		if (weight_)
			caffe_gpu_mul(n_, bottom_diff, bottom[2]->gpu_data(), bottom_diff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(BcoderLossLayer);

}  // namespace caffe
