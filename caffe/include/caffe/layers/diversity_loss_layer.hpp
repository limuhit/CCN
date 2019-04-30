#ifndef CAFFE_DIVERSITY_LOSS_LAYER_HPP_
#define CAFFE_DIVERSITY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
namespace caffe {

	template <typename Dtype>
		class DiversityLossLayer : public LossLayer<Dtype> {
		public:
			explicit DiversityLossLayer(const LayerParameter& param)
				: LossLayer<Dtype>(param), diff_() {}
			virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
			virtual inline const char* type() const { return "DiversityLoss"; }
			virtual inline bool AllowForceBackward(const int bottom_index) const {
				return true;
			}
			virtual inline int ExactNumBottomBlobs() const { return 2; }
			virtual inline int ExactTopBlobs() const { return 1; }
			void set_std(Dtype *m, int nclass);
		protected:
			virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
			virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
			virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
			virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
			Blob<Dtype> mean_;
			Blob<Dtype> std_;
			Blob<Dtype> diff_;
			Blob<Dtype> ones;
			Blob<Dtype> sdiff_;
			Blob<Dtype> tmp_;
			int h_, w_, ch_, num_;

		};
}

#endif  