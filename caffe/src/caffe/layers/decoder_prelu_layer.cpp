#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/decoder_prelu_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void DecoderPreluLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int channels = bottom[0]->channels();
		this->blobs_.resize(1);
		this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
		index_ = 0;
	}
	template <typename Dtype>
	void DecoderPreluLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		top[0]->ReshapeLike(*bottom[0]);
		num_ = bottom[0]->num();
		ch_ = bottom[0]->channels();
		w_ = bottom[0]->width();
		h_ = bottom[0]->height();
		mod_ = w_ + h_ - 1;
		if (bottom[0] == top[0]) {
			bottom_memory_.ReshapeLike(*bottom[0]);
		}
	}
	template<typename Dtype>
	void decoder_prelu_forward_cpu_kernel(const int num, const Dtype * const input, const Dtype * const slope,
		Dtype * const output, const int h_start, const int h_len, const int channel, const int height,
		const int width,const int plan_sum) {
		for (int i = 0; i < num; i++) {
			int ph = h_start + i % h_len;
			int pw = plan_sum - ph;
			int pc = (i / h_len) % channel;
			int pn = i / h_len / channel;
			int idx = ((pn*channel + pc)*height + ph)*width + pw;
			output[idx] = std::max(input[idx], Dtype(0))
				+ slope[pc] * std::min(input[idx], Dtype(0));
		}
	}
	template<typename Dtype>
	void decoder_prelu_copy_forward_cpu_kernel(const int num, const Dtype * const input,
		Dtype * const output, const int h_start, const int h_len, const int channel, const int height,
		const int width, const int plan_sum) {
		for (int i = 0; i < num; i++) {
			int ph = h_start + i % h_len;
			int pw = plan_sum - ph;
			int pc = (i / h_len) % channel;
			int pn = i / h_len / channel;
			int idx = ((pn*channel + pc)*height + ph)*width + pw;
			output[idx] = input[idx];
		}
	}
	template <typename Dtype>
	void DecoderPreluLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* slope_data = this->blobs_[0]->cpu_data();
		int plan_sum = index_;
		index_ = (index_ + 1) % mod_;
		int h_st = (plan_sum >= w_ ? plan_sum - w_ + 1 : 0);
		int h_end = (plan_sum >= h_ ? h_ : plan_sum + 1);
		int h_len = h_end - h_st;
		if (bottom[0] == top[0]) {
			decoder_prelu_copy_forward_cpu_kernel<Dtype>(h_len*ch_*num_, bottom_data, bottom_memory_.mutable_cpu_data(),
				h_st,h_len,ch_,h_,w_,plan_sum);
		}
		decoder_prelu_forward_cpu_kernel<Dtype>(h_len*ch_*num_, bottom_data, slope_data, top_data, h_st,
			h_len, ch_, h_, w_, plan_sum);
	}
	template<typename Dtype>
	void decoder_prelu_backward_cpu_kernel(const int num, const Dtype * const top_diff, const Dtype * const slope,
		const Dtype * const bottom_data, Dtype * const bottom_diff, const int inner_shape, const int channel) {
		for (int i = 0; i < num; i++) {
			int c = (i / inner_shape) % channel;
			bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
				+ slope[c] * (bottom_data[i] <= 0));
		}
	}
	template <typename Dtype>
	void DecoderPreluLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* slope_data = this->blobs_[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		const int count = bottom[0]->count();
		if (top[0] == bottom[0])
			bottom_data = bottom_memory_.cpu_data();
		decoder_prelu_backward_cpu_kernel<Dtype>(count, top_diff, slope_data, bottom_data,
			bottom[0]->mutable_cpu_diff(),h_*w_,ch_);
	}

#ifdef CPU_ONLY
	STUB_GPU(DecoderPreluLayer);
#endif

	INSTANTIATE_CLASS(DecoderPreluLayer);
	REGISTER_LAYER_CLASS(DecoderPrelu);

}  // namespace caffe
