#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/const_scale_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void ConstScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * const top_data = top[0]->mutable_gpu_data();
		const Dtype * const bottom_data = bottom[0]->gpu_data();
		int count = bottom[0]->count();
		caffe_gpu_scale(count, scale_, bottom_data, top_data);
		caffe_gpu_add_scalar(count, bias_, top_data);
	}
	template <typename Dtype>
	void ConstScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype * const top_diff = top[0]->gpu_diff();
		Dtype * const bottom_diff = bottom[0]->mutable_gpu_diff();
		int count = top[0]->count();
		caffe_gpu_scale(count, scale_, top_diff, bottom_diff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ConstScaleLayer);

}  // namespace caffe
