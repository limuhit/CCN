#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/split_channel_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void SplitChannelLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		channel_out_ = this->layer_param_.memory_data_param().channels();
	}
	template <typename Dtype>
	void SplitChannelLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		int n, h, w;
		n = bottom[0]->num();
		h = bottom[0]->height();
		w = bottom[0]->width();
		top[0]->Reshape(n,channel_out_,h,w);
	}
	template <typename Dtype>
	void split_data_forward_cpu_kernel(const int num, const Dtype* const bottom, Dtype * const top,
		const int channel_out, const int channel, const int inner_shape) {
		for (int i = 0; i < num; i++) {
			int ps = i % inner_shape;
			int pc = (i / inner_shape) % channel_out;
			int pn = i / inner_shape / channel_out;
			int pidx = (pn*channel + pc)*inner_shape + ps;
			top[i] = bottom[pidx];
		}
	}
	template <typename Dtype>
	void split_data_backward_cpu_kernel(const int num, Dtype* const bottom, const Dtype * const top,
		const int channel_out, const int channel, const int inner_shape) {
		for (int i = 0; i < num; i++) {
			int ps = i % inner_shape;
			int pc = (i / inner_shape) % channel_out;
			int pn = i / inner_shape / channel_out;
			int pidx = (pn*channel + pc)*inner_shape + ps;
			bottom[pidx] = top[i];
		}
	}
	template <typename Dtype>
	void SplitChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * const top_data = top[0]->mutable_cpu_data();
		const Dtype * const bottom_data = bottom[0]->cpu_data();
		split_data_forward_cpu_kernel<Dtype>(top[0]->count(), bottom_data, top_data, channel_out_, bottom[0]->channels(), bottom[0]->count(2));
	}

	template <typename Dtype>
	void SplitChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype * const bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype * const top_diff = top[0]->cpu_diff();
		caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
		split_data_backward_cpu_kernel<Dtype>(top[0]->count(), bottom_diff, top_diff, channel_out_, bottom[0]->channels(), bottom[0]->count(2));
	}

#ifdef CPU_ONLY
	STUB_GPU(SplitChannelLayer);
#endif

	INSTANTIATE_CLASS(SplitChannelLayer);
	REGISTER_LAYER_CLASS(SplitChannel);

}  // namespace caffe
