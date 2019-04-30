#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel/group_in;
			int tn = index / sz / sz / channel/group_out;
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
			if(tw+th+tc>=tn+sz-1)
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
			if (tw + th + tc > tn+sz-1)
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
	__global__ void conv_constrains_weight_kernel_v9(const int nthreads, Dtype* const weight,const int sz) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			if (tw + th>= sz - 1)
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v10(const int nthreads, Dtype* const weight,const int sz) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			if (tw + th >  sz - 1)
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v11(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out, const int *sc) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel;
			int tn = index / sz / sz / channel;
			if (tw + th  > sz - 1)
				weight[index] = 0;
			if ((tw + th == sz - 1) && (tc / group_in >= sc[tn]))
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v13(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			//int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (th+tc >= tn+sz/2)
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v14(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			//int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (th + tc > tn + sz / 2)
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v15(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tcg = (index / sz / sz) % channel % group_in /(group_in/4);
			int tn = index / sz / sz / channel / group_out;
			int tng = index / sz / sz / channel % group_out / (group_out / 4);
			if (th + tw + tc > tn + sz - 1)
				weight[index] = 0;
			else if (th + tw + tc == tn + sz - 1) {
				if (tng == 0)
					weight[index] = 0;
				else if (tng == 1 || tng == 2) {
					if (tcg >= 1)
						weight[index] = 0;		
				}
				else {
					if (tcg == 3)
						weight[index] = 0;
				}
			}
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v16(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tcg = (index / sz / sz) % channel % group_in / (group_in / 4);
			int tn = index / sz / sz / channel / group_out;
			int tng = index / sz / sz / channel % group_out / (group_out / 4);
			if (th + tw + tc > tn + sz - 1)
				weight[index] = 0;
			else if (th + tw + tc == tn + sz - 1) {
				if (tng == 0) {
					if(tcg>0)
						weight[index] = 0;
				}	
				else if (tng == 1 || tng == 2) {
					if (tcg > 2)
						weight[index] = 0;
				}
			}
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v17(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tn <= tc)
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_share_kernel(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out,const int num) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel;
			int skip = group_out*channel*sz*sz+group_in*sz*sz;
			Dtype * base = weight + channel*sz*sz*(num - group_out)+index;
			Dtype val = base[0];
			for (int i = 0; i < tc; i++) {
				base -= skip;
				val+=base[0];
			}
			val = val / (tc + 1);
			for (int i = 0; i <= tc; i++) {
				base[0] = val;
				base += skip;
			}
		}
	}
__global__ void sync_conv_groups() { }
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	if(this->constrains_==7 &&(!this->copy_finished_)){
		caffe_gpu_memcpy(this->copy_weight_.count()*sizeof(Dtype), this->blobs_[0]->gpu_data(), this->copy_weight_.mutable_gpu_data());
		caffe_gpu_memcpy(this->copy_bias_.count() * sizeof(Dtype), this->blobs_[1]->gpu_data(), this->copy_bias_.mutable_gpu_data());
		this->copy_finished_ = true;
	}
	if ((this->constrains_ == 11 || this->constrains_ == 12) && !this->constrain_init_) {
		int * mt = this->constrain_tmp_.mutable_cpu_data();
		int ch = this->blobs_[0]->channels();
		int ngroup = ch / group_in_;
		int ws = sqrt(ngroup);
		int ac = 0;
		if (constrains_ == 11) {
			for (int i = 1; i <= ws; i++) {
				
				for (int j = 0; j < i*group_out_; j++) 	mt[j] = ac ;
				mt = mt + i*group_out_;
				ac += i;
			}
			for (int i = ws - 1; i > 0; i--) {
				
				for (int j = 0; j < i*group_out_; j++) 	mt[j] = ac ;
				mt = mt + i*group_out_;
				ac += i;
			}
		}
		else {
			for (int i = 1; i <= ws; i++) {
				ac += i;
				for (int j = 0; j < i*group_out_; j++) 	mt[j] = ac;
				mt = mt + i*group_out_;
			}
			for (int i = ws - 1; i > 0; i--) {
				ac += i;
				for (int j = 0; j < i*group_out_; j++) 	mt[j] = ac;
				mt = mt + i*group_out_;
			}
		}
		mt = this->constrain_tmp_.mutable_cpu_data();
		this->constrain_init_ = true;
	}
	//LOG(INFO) << "used cudnn";
	if (share_filter_) {
		int n = this->blobs_[0]->num();
		int ch = this->blobs_[0]->channels();
		int sz = this->blobs_[0]->width();
		conv_constrains_share_kernel<Dtype> << <CAFFE_GET_BLOCKS(ch*sz*sz*group_out_), CAFFE_CUDA_NUM_THREADS >> >
			(ch*sz*sz*group_out_, this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_,n);
	}
	if (constrains_>0){
		int n = this->blobs_[0]->num();
		int ch = this->blobs_[0]->channels();
		int sz = this->blobs_[0]->width();
		if (constrains_ == 1) {
			conv_constrains_weight_kernel<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
			//LOG(INFO) << "constarin1";
		}
		else if(constrains_==2){
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
				caffe_gpu_memcpy(channel_per_group*this->freeze_idx_*this->copy_weight_.count(1)*sizeof(Dtype), cp_w, tp_w);
				caffe_gpu_memcpy(channel_per_group*this->freeze_idx_*this->copy_bias_.count(1)*sizeof(Dtype), cp_b, tp_b);
			}
			tp_w = tp_w + (this->freeze_idx_ + 1)*this->copy_weight_.count(1)*channel_per_group;
			tp_b = tp_b + (this->freeze_idx_ + 1)*this->copy_bias_.count(1)*channel_per_group;
			if (this->freeze_idx_ < this->group_ - 1) {
				int base_n = channel_per_group*(this->group_ - 1 - this->freeze_idx_);
				caffe_gpu_set<Dtype>(base_n*this->copy_weight_.count(1), Dtype(0), tp_w);
				caffe_gpu_set<Dtype>(base_n*this->copy_bias_.count(1), Dtype(0), tp_b);
			}
			

		}
		else if (constrains_ == 9) {//only used for group convolution and for single channel image prediction
			conv_constrains_weight_kernel_v9<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), sz);
		}
		else if (constrains_ == 10) {//only used for group convolution and for single channel image prediction
			conv_constrains_weight_kernel_v10<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), sz);
		}
		else if (constrains_ == 11 || constrains_ == 12) {//only used for group convolution and for single channel image prediction
			const int * ts = this->constrain_tmp_.gpu_data();
			conv_constrains_weight_kernel_v11<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(),ch,sz,group_in_,group_out_,ts);
		}
		else if (constrains_ == 13) {
			conv_constrains_weight_kernel_v13<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
		}
		else if (constrains_ == 14) {
			conv_constrains_weight_kernel_v14<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
		}
		else if (constrains_ == 15) {
			conv_constrains_weight_kernel_v15<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
		}
		else if (constrains_ == 16) {
			conv_constrains_weight_kernel_v16<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
		}
		else if (constrains_ == 17) {
			conv_constrains_weight_kernel_v17<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
		}
		CUDA_POST_KERNEL_CHECK;
	}
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!conv_need_backward_){
		return;
	}
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
