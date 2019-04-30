#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/decoder_extract_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void DecoderExtractLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		DecoderParameter rm = this->layer_param_.decoder_param();
		label_ = rm.extract_label();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		channel_ = bottom[0]->channels();
		num_ = bottom[0]->num();
		top[0]->Reshape(num_,1, height_, width_);
		init_index();
	}
	template <typename Dtype>
	void DecoderExtractLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		if ((height_ != bottom[0]->height()) || (width_ != bottom[0]->width()) || (channel_ != bottom[0]->channels()) || (num_ != bottom[0]->num())) {
			height_ = bottom[0]->height();
			width_ = bottom[0]->width();
			channel_ = bottom[0]->channels();
			num_ = bottom[0]->num();
			init_index();
		}
		top[0]->Reshape(num_, 1, height_, width_);
	}
	template <typename Dtype>
	void DecoderExtractLayer<Dtype>::init_index() {
		pidx_ = 0;
		index_.Reshape(1, 1, height_, width_);
		mod_ = height_ + width_ + channel_;
		int * idx = index_.mutable_cpu_data();
		start_idx_.clear();
		int index = 0;
		for (int ps = 0; ps < height_ + width_ - 1; ps++) {
			start_idx_.push_back(index);
			for (int i = 0; i < height_; i++) {
				int j = ps - i;
				if (j < 0 || j >= width_)
					continue;
				idx[index] = i*width_ + j;
				index++;
			}
		}
		start_idx_.push_back(index);
	}
	template <typename Dtype>
	void DecoderExtractLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	}
	template <typename Dtype>
	void DecoderExtractLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


	}

#ifdef CPU_ONLY
	STUB_GPU(DecoderExtract);
#endif

	INSTANTIATE_CLASS(DecoderExtractLayer);
	REGISTER_LAYER_CLASS(DecoderExtract);

}  // namespace caffe
