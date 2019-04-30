#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/decoder_output_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void DecoderOutputLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		DecoderParameter rm = this->layer_param_.decoder_param();
		total_region_ = static_cast<Dtype>(rm.total_region());
		nchannel_ = rm.nchannels();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		channel_ = bottom[0]->channels();
		num_ = bottom[0]->num();
		ngroup_ = channel_ / nchannel_;
		top[0]->Reshape(num_ * height_ * width_ + 1, 1, 1, ngroup_+1);
		tmp_.Reshape(num_*height_*width_, 1, 1, ngroup_);
		csum_.Reshape(num_*height_*width_, 1, 1, 1);
		cmax_.Reshape(num_*height_*width_, 1, 1, 1);
		init_index();
	}
	template <typename Dtype>
	void DecoderOutputLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		if ((height_ != bottom[0]->height()) || (width_ != bottom[0]->width()) || (channel_ != bottom[0]->channels()) || num_ != bottom[0]->num()) {
			height_ = bottom[0]->height();
			width_ = bottom[0]->width();
			channel_ = bottom[0]->channels();
			num_ = bottom[0]->num();
			ngroup_ = channel_ / nchannel_;
			top[0]->Reshape(num_ * height_ * width_+1, 1,1, ngroup_+1);
			tmp_.Reshape(num_*height_*width_, 1, 1, ngroup_);
			csum_.Reshape(num_*height_*width_, 1, 1, 1);
			cmax_.Reshape(num_*height_*width_, 1, 1, 1);
			init_index();
		}

	}
	template <typename Dtype>
	void DecoderOutputLayer<Dtype>::init_index() {
		pidx_ = 0;
		mod_ = height_ + width_ + nchannel_;
		index_.Reshape(1, 1, height_, width_);
		int * idx = index_.mutable_cpu_data();
		start_idx_.clear();
		int index = 0;
		for (int ps = 0; ps < height_+width_-1; ps++) {
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
	void DecoderOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		
	}
	template <typename Dtype>
	void DecoderOutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


	}

#ifdef CPU_ONLY
	STUB_GPU(DecoderOutput);
#endif

	INSTANTIATE_CLASS(DecoderOutputLayer);
	REGISTER_LAYER_CLASS(DecoderOutput);

}  // namespace caffe
