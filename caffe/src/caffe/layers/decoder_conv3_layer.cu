#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/decoder_conv3_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void deocder_conv3_data_to_col_gpu(const int size, const Dtype * input, 	Dtype * output, const int * index, const int index_stride,
		const int * inv_index,const int kernel_size, const int rchannel, const int height, const int width, 
	    const int lw, const int start_idx, const int start_sum,const int innershape, 
		const int num, const int channel) {
			CUDA_KERNEL_LOOP(i, size) {
				int pw = i % kernel_size;
				int ph = (i / kernel_size) % kernel_size;
				int pc = (i / kernel_size / kernel_size) % rchannel;
				int pn = (i / kernel_size/ kernel_size / rchannel) % num;
				int pb = i / kernel_size / kernel_size / rchannel / num;
				int ps = i % innershape;
				int pidx = pb + start_idx;
				int th = index[pidx];
				int tw = index[pidx + index_stride];
				int lidx = inv_index[th*width + tw];
				int out_idx = (((th + tw - start_sum)*lw + lidx)*num+pn)*innershape+ps;
				int bh = th + ph - kernel_size / 2;
				int bw = tw + pw - kernel_size / 2;
				//output[i] = tw;
				if (bh < 0 || bh >= height || bw < 0 || bw >= width)
				{
					output[out_idx] = 0;
					continue;
				}
				int in_idx = ((pn*channel+pc)*height + bh)*width + bw;
				output[out_idx] = input[in_idx];
			}
		}
	template <typename Dtype>
	__global__ void deocder_conv3_col_to_data_gpu(const int size, const Dtype * input, Dtype * output, const int * index, const int index_stride,
		const int * inv_index, const int group_out, const int lw, const int start_idx, const int start_sum,  const int psum, 
		const int height, const int width, const int nout, const int num, const Dtype * bias) {
		CUDA_KERNEL_LOOP(i, size) {
			int ps = i % group_out;
			int pb = ( i / group_out ) / num;
			int pn = ( i / group_out ) % num;
			int pidx = pb + start_idx;
			int th = index[pidx];
			int tw = index[pidx + index_stride];
			int lidx = inv_index[th*width + tw];
			int in_idx = (((th + tw - start_sum)*lw + lidx)*num + pn)*group_out + ps;
			int pc = psum - th - tw;
			int out_idx = ((pn*nout+pc*group_out + ps)*height + th)*width + tw;
			int bias_idx = pc*group_out + ps;
			output[out_idx] = input[in_idx]+bias[bias_idx];
		}
	}
	template <typename Dtype>
	__global__ void conv3_constrains_weight_kernel_v5(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tw + th + tc >= tn + sz - 1)
				weight[index] = Dtype(0);
		}
	}
	template <typename Dtype>
	__global__ void conv3_constrains_weight_kernel_v6(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tw + th + tc > tn + sz - 1)
				weight[index] = Dtype(0);
		}
	}
	template <typename Dtype>
	__global__ void conv3_transpose_kernel(const int nthreads, const Dtype * const input,Dtype* const output,
		const int inner_shape, const int ngroup) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int ts = index % inner_shape;
			int tn = index / inner_shape;
			int tidx = (ngroup-tn-1)*inner_shape + ts;
			output[tidx] = input[index];
		}
	}
	template <typename Dtype>
	void DecoderConv3Layer<Dtype>::init_weight() {
		if (constrain_ == 5) {
			conv3_constrains_weight_kernel_v5<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch_, kernel_size_, group_in_, group_out_);
		}
		else if (constrain_ == 6) {
			conv3_constrains_weight_kernel_v6<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch_, kernel_size_, group_in_, group_out_);

		}
		weight_init_ = true;

		int inner_shape = group_out_*ch_*kernel_size_*kernel_size_;
		conv3_transpose_kernel<Dtype> << <CAFFE_GET_BLOCKS(inner_shape*ngroup_), CAFFE_CUDA_NUM_THREADS >> >
			(inner_shape*ngroup_, this->blobs_[0]->gpu_data(), weight_.mutable_gpu_data(), inner_shape, ngroup_);

	}
	template <typename Dtype>
	void DecoderConv3Layer<Dtype>::single_forward(int la, int lb, int psum, const Dtype * bottom, Dtype * top) {
		int min_t = h_ > w_ ? w_ : h_;
		int max_t = h_ > w_ ? h_ : w_;
		int lw = plan_idx_[lb + 1] - plan_idx_[lb];
		if (la <= max_t-1 && lb >= min_t-1) {
			lw = min_t;
		}
		else if (la >= max_t) {
			lw = plan_idx_[la + 1] - plan_idx_[la];
		}
		int ch = psum - la + kernel_size_ > ngroup_ ? ngroup_*group_in_ : (psum - la + kernel_size_)*group_in_;
		int inner_shape = ch*kernel_size_*kernel_size_;
		int cnt = num_*(plan_idx_[lb + 1] - plan_idx_[la])*inner_shape;
		//LOG(INFO) <<psum<<" "<< lw << " " << la;
		caffe_gpu_set(num_*lw*(lb - la + 1)*inner_shape, Dtype(0), tmp_.mutable_gpu_data());
		deocder_conv3_data_to_col_gpu<Dtype><<<CAFFE_GET_BLOCKS(cnt),CAFFE_CUDA_NUM_THREADS>>>(cnt, bottom, tmp_.mutable_gpu_data(), index_.gpu_data(), h_*w_,
			inv_index_.gpu_data(), kernel_size_, ch,  h_, w_,  lw, plan_idx_[la], la, inner_shape,num_,ch_);
		
		const Dtype * weight = weight_.gpu_data();
		int weight_shape = group_out_*ch_*kernel_size_*kernel_size_;
		weight = weight + (ngroup_- psum + la - 1)*weight_shape;
		caffe_gpu_gemm_batch(CblasNoTrans, CblasTrans, lw*num_, group_out_, inner_shape, Dtype(1.0),
			tmp_.gpu_data(), weight, Dtype(0), res_.mutable_gpu_data(), lw*num_*inner_shape, weight_shape,
			lw*num_*group_out_, lb-la+1,inner_shape, ch_*kernel_size_*kernel_size_);
		cnt = (plan_idx_[lb + 1] - plan_idx_[la])*group_out_*num_;
		const Dtype * bias = this->blobs_[1]->gpu_data();
		deocder_conv3_col_to_data_gpu<Dtype> << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >(cnt, res_.gpu_data(), top, index_.gpu_data(), h_*w_,
			inv_index_.gpu_data(), group_out_, lw, plan_idx_[la], la, psum,	h_, w_, nout_,num_,bias);
	}
	template <typename Dtype>
	void DecoderConv3Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* bias = this->blobs_[1]->gpu_data();
		Dtype * res = res_.mutable_gpu_data();
		int plan_sum = pindex_;
		if (!weight_init_)
			init_weight();
		const Dtype* weight = weight_.gpu_data();
		pindex_ = (pindex_ + 1) % mod_;
		int la = plan_sum >= ngroup_ ? plan_sum - ngroup_ + 1 : 0;
		int lb = plan_sum > h_ + w_ - 2 ? h_ + w_ - 2 : plan_sum;
		int one_third = (h_ + w_) / 3;
		int two_third = one_third * 2;
	
		if (la < one_third) {
				if (lb < one_third) {
					single_forward(la, lb, plan_sum, bottom_data, top_data);
					//caffe_gpu_memcpy(this->blobs_[2]->count() * sizeof(Dtype), tmp_.gpu_data(), this->blobs_[2]->mutable_gpu_data());
				}
				else if (lb < two_third) {
					single_forward(la, one_third, plan_sum, bottom_data, top_data);
					//caffe_gpu_memcpy(this->blobs_[2]->count() * sizeof(Dtype), tmp_.gpu_data(), this->blobs_[2]->mutable_gpu_data());
					single_forward(one_third, lb, plan_sum, bottom_data, top_data);
				}
				else {
					single_forward(la, one_third, plan_sum, bottom_data, top_data);
					//caffe_gpu_memcpy(this->blobs_[2]->count() * sizeof(Dtype), tmp_.gpu_data(), this->blobs_[2]->mutable_gpu_data());
					single_forward(one_third, two_third, plan_sum, bottom_data, top_data);
					single_forward(two_third, lb, plan_sum, bottom_data, top_data);
				}

			}
		else if (la < two_third) {
				if (lb < two_third) {
					single_forward(la, lb, plan_sum, bottom_data, top_data);
					//caffe_gpu_memcpy(this->blobs_[2]->count() * sizeof(Dtype), tmp_.gpu_data(), this->blobs_[2]->mutable_gpu_data());
				}
				else {
					single_forward(la, two_third, plan_sum, bottom_data, top_data);
					//caffe_gpu_memcpy(this->blobs_[2]->count() * sizeof(Dtype), tmp_.gpu_data(), this->blobs_[2]->mutable_gpu_data());
					single_forward(two_third, lb, plan_sum, bottom_data, top_data);
				}
			}
		else {
				single_forward(la, lb, plan_sum, bottom_data, top_data);
				//caffe_gpu_memcpy(this->blobs_[2]->count() * sizeof(Dtype), tmp_.gpu_data(), this->blobs_[2]->mutable_gpu_data());
			}

		CUDA_POST_KERNEL_CHECK;
			//bottom_data += (ch_*h_*w_);
			//top_data += (nout_*h_*w_);
		//}	
	}
	template <typename Dtype>
	void DecoderConv3Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DecoderConv3Layer);

}  // namespace caffe
