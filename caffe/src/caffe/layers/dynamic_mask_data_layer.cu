#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dynamic_mask_data_layer.hpp"
namespace caffe {


	template <typename Dtype>
	void DynamicMaskDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		top[0]->ShareData(data_);
	}

	template <typename Dtype>
	void DynamicMaskDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

	INSTANTIATE_LAYER_GPU_FUNCS(DynamicMaskDataLayer);

}  // namespace caffe
