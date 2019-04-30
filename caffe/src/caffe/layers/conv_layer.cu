#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tn > tc) continue;
			if (tn == tc) {
				if (th < sz / 2) continue;
				else if (th == sz / 2) {
					if (tw < sz / 2)continue;
					else weight[index] = 0;
				}
				else
					weight[index] = 0;
			}
			else
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v2(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tn > tc) continue;
			if (tn == tc) {
				if (th < sz / 2) continue;
				else if (th == sz / 2) {
					if (tw <= sz / 2)continue;
					else weight[index] = 0;
				}
				else
					weight[index] = 0;
			}
			else
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v3(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tn > tc) {
				if (th <= sz / 2) {
					if (tw <= sz / 2)continue;
					else weight[index] = 0;
				}
				else
					weight[index] = 0;
			}
			else if (tn == tc) {
				if (th < sz / 2) {
					if (tw <= sz / 2)continue;
					else weight[index] = 0;
				}
				else if (th == sz / 2) {
					if (tw < sz / 2)continue;
					else weight[index] = 0;
				}
				else
					weight[index] = 0;
			}
			else
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v4(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tn >= tc) {
				if (th <= sz / 2) {
					if (tw <= sz / 2)continue;
					else weight[index] = 0;
				}
				else
					weight[index] = 0;
			}
			else
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v5(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tw + th + tc >= tn + sz - 1)
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v6(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tw + th + tc > tn + sz - 1)
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v7(const int nthreads, Dtype* const weight, const Dtype* const copy,
		const int freeze, const int inner_size) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tn = index / inner_size;
			if (tn < freeze)
				weight[index] = copy[index];
			else if (tn > freeze)
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_share_kernel(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out, const int num) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel;
			int skip = group_out*channel*sz*sz + group_in*sz*sz;
			Dtype * base = weight + channel*sz*sz*(num - group_out) + index;
			Dtype val = base[0];
			for (int i = 0; i < tc; i++) {
				base -= skip;
				val += base[0];
			}
			val = val / (tc + 1);
			for (int i = 0; i <= tc; i++) {
				base[0] = val;
				base += skip;
			}
		}
	}
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->constrains_ == 7 && (!this->copy_finished_)) {
		caffe_gpu_memcpy(this->copy_weight_.count() * sizeof(Dtype), this->blobs_[0]->gpu_data(), this->copy_weight_.mutable_gpu_data());
		caffe_gpu_memcpy(this->copy_bias_.count() * sizeof(Dtype), this->blobs_[1]->gpu_data(), this->copy_bias_.mutable_gpu_data());
		this->copy_finished_ = true;
	}
  //LOG(INFO) << "used cudnn";
  if (share_filter_) {
		int n = this->blobs_[0]->num();
		int ch = this->blobs_[0]->channels();
		int sz = this->blobs_[0]->width();
		conv_constrains_share_kernel<Dtype> << <CAFFE_GET_BLOCKS(ch*sz*sz*group_out_), CAFFE_CUDA_NUM_THREADS >> >
			(ch*sz*sz*group_out_, this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_, n);
	}
  if (constrains_>0) {
		int n = this->blobs_[0]->num();
		int ch = this->blobs_[0]->channels();
		int sz = this->blobs_[0]->width();
		if (constrains_ == 1) {
			conv_constrains_weight_kernel<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
			//LOG(INFO) << "constarin1";
		}
		else if (constrains_ == 2) {
			conv_constrains_weight_kernel_v2<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
			//LOG(INFO) << "constarin2";
		}
		else if (constrains_ == 3) {
			conv_constrains_weight_kernel_v3<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
		}
		else if (constrains_ == 4) {
			conv_constrains_weight_kernel_v4<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
		}
		else if (constrains_ == 5) {
			conv_constrains_weight_kernel_v5<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
		}
		else if (constrains_ == 6) {
			conv_constrains_weight_kernel_v6<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);

		}
		else if (constrains_ == 7) {
			Dtype * tp_w = this->blobs_[0]->mutable_gpu_data();
			Dtype * tp_b = this->blobs_[1]->mutable_gpu_data();
			const Dtype * const cp_w = this->copy_weight_.gpu_data();
			const Dtype * const cp_b = this->copy_bias_.gpu_data();
			int nchannel = this->copy_weight_.num();
			int channel_per_group = nchannel / this->group_;
			if (this->freeze_idx_ > 0) {
				caffe_gpu_memcpy(channel_per_group*this->freeze_idx_*this->copy_weight_.count(1) * sizeof(Dtype), cp_w, tp_w);
				caffe_gpu_memcpy(channel_per_group*this->freeze_idx_*this->copy_bias_.count(1) * sizeof(Dtype), cp_b, tp_b);
			}
			tp_w = tp_w + (this->freeze_idx_ + 1)*this->copy_weight_.count(1)*channel_per_group;
			tp_b = tp_b + (this->freeze_idx_ + 1)*this->copy_bias_.count(1)*channel_per_group;
			if (this->freeze_idx_ < this->group_ - 1) {
				int base_n = channel_per_group*(this->group_ - 1 - this->freeze_idx_);
				caffe_gpu_set<Dtype>(base_n*this->copy_weight_.count(1), Dtype(0), tp_w);
				caffe_gpu_set<Dtype>(base_n*this->copy_bias_.count(1), Dtype(0), tp_b);
			}


		}

		CUDA_POST_KERNEL_CHECK;
	}
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!conv_need_backward_){ return; }
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
