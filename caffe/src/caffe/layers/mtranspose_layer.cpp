#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mtranspose_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void MTransposeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		MTransposeParameter mp = this->layer_param_.mtranspose_param();
		groups_ = mp.groups();
		inverse_ = mp.inverse();
		if (inverse_) {
			width_ = mp.width();
			height_ = mp.height();
			channel_ = mp.channel();
			num_ = bottom[0]->count() / width_ / height_ / channel_;
			inner_size_ = width_*height_;
			top[0]->Reshape(num_, channel_, height_, width_);
		}
		else {
			width_ = bottom[0]->width();
			height_ = bottom[0]->height();
			channel_ = bottom[0]->channels();
			num_ = bottom[0]->num();
			inner_size_ = width_*height_;
			vector<int> shape;
			shape.push_back(num_*channel_*height_*width_ / groups_);
			shape.push_back(groups_);
			top[0]->Reshape(shape);
		}
	}
	template <typename Dtype>
	void MTransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (inverse_) {
			num_ = bottom[0]->count() / width_ / height_ / channel_;
			top[0]->Reshape(num_, channel_, height_, width_);
		}
		else {
			width_ = bottom[0]->width();
			height_ = bottom[0]->height();
			channel_ = bottom[0]->channels();
			num_ = bottom[0]->num();
			inner_size_ = width_*height_;
			vector<int> shape;
			shape.push_back(num_*channel_*height_*width_ / groups_);
			shape.push_back(groups_);
			top[0]->Reshape(shape);
		}
	}
	template <typename Dtype>
	void mtranspose_forward_cpu_kernel(const int num, const Dtype * const bottom, Dtype * const top,
		const int inner_size, const int channel, const int groups) {
		for (int i = 0; i < num; i++) {
			int pn = i / inner_size / channel;
			int pc = (i / inner_size) % channel;
			int ps = i % inner_size;
			int tidx = (pn*inner_size*channel/groups + pc/groups*inner_size+ps)*groups+pc%groups;
			top[tidx] = bottom[i];
		}
	}
	template <typename Dtype>
	void mtranspose_forward_cpu_kernel_inverse(const int num, const Dtype * const bottom, Dtype * const top,
		const int inner_size, const int channel, const int groups) {
		for (int i = 0; i < num; i++) {
			int pn = i / inner_size / channel;
			int pc = (i / inner_size) % channel;
			int ps = i % inner_size;
			int tidx = (pn*inner_size*channel / groups + pc / groups*inner_size + ps)*groups + pc%groups;
			//top[tidx] = bottom[i];
			top[i] = bottom[tidx];
		}
	}
	template <typename Dtype>
	void MTransposeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		if (inverse_) {
			mtranspose_forward_cpu_kernel_inverse<Dtype>(bottom[0]->count(), bottom_data,
				top[0]->mutable_cpu_data(), inner_size_, channel_, groups_);
		}
		else {
			mtranspose_forward_cpu_kernel<Dtype>(bottom[0]->count(), bottom_data,
				top[0]->mutable_cpu_data(), inner_size_, channel_, groups_);
		}
		

	}
	template <typename Dtype>
	void mtranspose_backward_cpu_kernel(const int num, Dtype * const bottom, const Dtype * const top,
		const int inner_size, const int channel, const int groups) {
		for (int i = 0; i < num; i++) {
			int pn = i / inner_size / channel;
			int pc = (i / inner_size) % channel;
			int ps = i % inner_size;
			int tidx = (pn*inner_size*channel / groups + pc / groups*inner_size + ps)*groups + pc%groups;
			bottom[i]=top[tidx];
		}
	}
	template <typename Dtype>
	void MTransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (!inverse_) {
			mtranspose_backward_cpu_kernel<Dtype>(bottom[0]->count(), bottom_diff,
				top[0]->cpu_diff(), inner_size_, channel_, groups_);
		}
		
	}

#ifdef CPU_ONLY
	STUB_GPU(MTransposeLayer);
#endif

	INSTANTIATE_CLASS(MTransposeLayer);
	REGISTER_LAYER_CLASS(MTranspose);

}  // namespace caffe
