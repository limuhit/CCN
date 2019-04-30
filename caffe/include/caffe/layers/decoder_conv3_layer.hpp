#ifndef CAFFE_DECODER_CONV3_LAYER_HPP_
#define CAFFE_DECODER_CONV3_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {
	template <typename Dtype>
	class DecoderConv3Layer : public Layer<Dtype> {
	public:
		explicit DecoderConv3Layer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "DecoderConv3"; }
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int kernel_size_, w_, h_, ch_, num_, nout_;
		int constrain_, group_in_, group_out_,ngroup_;
		int mod_, pindex_;
		int mem_size_;
		Blob<Dtype> tmp_;
		Blob<Dtype> weight_;
		Blob<int> index_;
		Blob<int> inv_index_;
		vector<int> plan_idx_;
		int plan_kernel_size_;
		Blob<Dtype> res_;
		bool weight_init_;
		void init_index();
		void init_weight();
		void init_weight_cpu();
		void single_forward(int la, int lb, int psum, const Dtype * bottom, Dtype * top);
		void single_forward_cpu(int la, int lb, int psum, const Dtype * bottom, Dtype * top);
		
		
	};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_

