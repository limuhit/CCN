#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dynamic_mask_data_layer.hpp"
namespace caffe {
	template <typename Dtype>
	void DynamicMaskDataLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		this->blobs_.resize(1);
		this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
		channel_out_ = bottom[0]->channels();
		this->blobs_[0]->mutable_cpu_data()[0] = channel_out_;
		n_ = bottom[0]->num();
		c_ = bottom[0]->channels();
		h_ = bottom[0]->height();
		w_ = bottom[0]->width();
		data_.Reshape(n_, c_, h_, w_);
		set_data();
	}
	template <typename Dtype>
	void DynamicMaskDataLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		bool modify = false;
		int tchannel = int(this->blobs_[0]->cpu_data()[0] + 0.000001);
		if (n_*c_*h_*w_ != bottom[0]->count()|| tchannel!=channel_out_)
			modify = true;
		n_ = bottom[0]->num();
		c_ = bottom[0]->channels();
		h_ = bottom[0]->height();
		w_ = bottom[0]->width();
		top[0]->Reshape(n_, c_, h_, w_);
		data_.Reshape(n_, c_, h_, w_);
		if (modify) {
			channel_out_ = tchannel;
			set_data();
		}
	}
	template <typename Dtype>
	void DynamicMaskDataLayer<Dtype>::set_data() {
		Dtype * data = data_.mutable_cpu_data();
		int inner_size = c_*h_*w_;
		for (int i = 0; i < c_; i++) {
			if (i < channel_out_)
				caffe_set(h_*w_, Dtype(1.0), data + i*h_*w_);
			else
				caffe_set(h_*w_, Dtype(0), data + i*h_*w_);
		}
		for (int i = 1; i < n_; i++)
			caffe_copy(inner_size, data, data + i*inner_size);
	}
	template <typename Dtype>
	void DynamicMaskDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		top[0]->ShareData(data_);
	}
	template <typename Dtype>
	void  DynamicMaskDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
}
#ifdef CPU_ONLY
	STUB_GPU(DynamicMaskDataLayer);
#endif

	INSTANTIATE_CLASS(DynamicMaskDataLayer);
	REGISTER_LAYER_CLASS(DynamicMaskData);

}  // namespace caffe
