#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/decoder_quant_layer.hpp"
namespace caffe {
	
	template <typename Dtype>
	void DecoderQuantLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		MLQuantParameter rm = this->layer_param_.mlquant_param();
		levels_ = rm.groups();
		this->blobs_.resize(3);
		ch_ = bottom[0]->channels();
		this->blobs_[0].reset(new Blob<Dtype>(1, ch_, 1, levels_));
		this->blobs_[2].reset(new Blob<Dtype>(1, ch_, 1, levels_));	
		this->blobs_[1].reset(new Blob<Dtype>(5, 1, 1, 1));

	}
	template <typename Dtype>
	void DecoderQuantLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		top[0]->ReshapeLike(*bottom[0]);
		weight_.ReshapeLike(*this->blobs_[0]);
		h_ = bottom[0]->height();
		w_ = bottom[0]->width();
		ch_ = bottom[0]->channels();
		num_ = bottom[0]->num();
	}
	

	template <typename Dtype>
	void DecoderQuantLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		
	}
	
	template <typename Dtype>
	void DecoderQuantLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		

	}

#ifdef CPU_ONLY
	STUB_GPU(DecoderQuantLayer);
#endif

	INSTANTIATE_CLASS(DecoderQuantLayer);
	REGISTER_LAYER_CLASS(DecoderQuant);

}  // namespace caffe
