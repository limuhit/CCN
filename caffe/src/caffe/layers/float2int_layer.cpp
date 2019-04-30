#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/float2int_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void Float2IntLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		train_ = this->phase_ == TRAIN;
		Float2IntParameter fp = this->layer_param_.float2int_param();
		if (fp.quant())
			train_ = false;
	}
	template <typename Dtype>
	void Float2IntLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		top[0]->ReshapeLike(*bottom[0]);
		noise_.Reshape(bottom[0]->count(),1,1,1);
	}
	template <typename Dtype>
	void float2int_cpu_kernel(const int num, const Dtype* const bottom, Dtype * const top) {
		for (int i = 0; i < num; i++) {
			top[i] = static_cast<Dtype>(static_cast<int>(bottom[i]));
			if (top[i] < 0)top[i] = 0;
			if (top[i] > 255)top[i] = 255;
		}
	}
	template <typename Dtype>
	void Float2IntLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * const top_data = top[0]->mutable_cpu_data();
		const Dtype * const bottom_data = bottom[0]->cpu_data();
		float2int_cpu_kernel<Dtype>(bottom[0]->count(), bottom_data, top_data);
	}

	template <typename Dtype>
	void Float2IntLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
	}

#ifdef CPU_ONLY
	STUB_GPU(RoundLayer);
#endif

	INSTANTIATE_CLASS(Float2IntLayer);
	REGISTER_LAYER_CLASS(Float2Int);

}  // namespace caffe
