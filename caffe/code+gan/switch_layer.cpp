#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/switch_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void SwitchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		selected_ = 0;
	}
	template <typename Dtype>
	void SwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		top[0]->ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void SwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype *bottom_data = bottom[selected_]->cpu_data();
		Dtype *top_data = top[0]->mutable_cpu_data();
		caffe_copy(top[0]->count(), bottom_data, top_data);
		//LOG(INFO) << "select bottom " << sl[0];
	}
	template <typename Dtype>
	void SwitchLayer<Dtype>::select(int idx)
	{ 
		selected_ = idx;
		//LOG(INFO) << "select bottom " << selected_;
	}
	template <typename Dtype>
	void SwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
		Dtype *bottom_diff = bottom[selected_]->mutable_cpu_diff();
		const Dtype *top_diff = top[0]->cpu_diff();
		caffe_copy(top[0]->count(), top_diff, bottom_diff);
	}


#ifdef CPU_ONLY
	STUB_GPU(SwitchLayer);
#endif

	INSTANTIATE_CLASS(SwitchLayer);
	REGISTER_LAYER_CLASS(Switch);

}  // namespace caffe
