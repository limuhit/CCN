
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/switch_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void SwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		//top[0]->ShareData(*bottom[selected_]);
		const Dtype *bottom_data = bottom[selected_]->gpu_data();
		Dtype *top_data = top[0]->mutable_gpu_data();
		caffe_copy(top[0]->count(), bottom_data, top_data);
	}

	template <typename Dtype>
	void SwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		//bottom[selected_]->ShareDiff(*top[0]);
		Dtype *bottom_diff = bottom[selected_]->mutable_gpu_diff();
		const Dtype *top_diff = top[0]->gpu_diff();
		caffe_copy(top[0]->count(), top_diff, bottom_diff);
	}


	INSTANTIATE_LAYER_GPU_FUNCS(SwitchLayer);

}  // namespace caffe
