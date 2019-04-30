#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/const_scale_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void ConstScaleLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		ConstScaleParameter rm = this->layer_param_.const_scale_param();
		scale_ = rm.scale();
		bias_ = rm.bias();
		top[0]->ReshapeLike(*bottom[0]);
	}
	template <typename Dtype>
	void ConstScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * const top_data = top[0]->mutable_cpu_data();
		const Dtype * const bottom_data = bottom[0]->cpu_data();
		int count = bottom[0]->count();
		caffe_cpu_scale(count, scale_, bottom_data, top_data);
		caffe_add_scalar(count, bias_, top_data);
	}

	template <typename Dtype>
	void ConstScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype * const top_diff = top[0]->cpu_diff();
		Dtype * const bottom_diff = bottom[0]->mutable_cpu_diff();
		int count = top[0]->count();
		caffe_cpu_scale(count, scale_, top_diff, bottom_diff);

	}

#ifdef CPU_ONLY
	STUB_GPU(ConstScaleLayer);
#endif

	INSTANTIATE_CLASS(ConstScaleLayer);
	REGISTER_LAYER_CLASS(ConstScale);

}  // namespace caffe
