#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/decoder_conv3_layer.hpp"
//#include "mkl.h"
namespace caffe {

	template <typename Dtype>
	void DecoderConv3Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const ConvolutionParameter& param = this->layer_param_.convolution_param();
		constrain_ = param.constrain();
		group_in_ = param.group_in();
		group_out_ = param.group_out();
		kernel_size_ = param.kernel_size(0);
		nout_ = param.num_output();
		h_ = bottom[0]->height();
		w_ = bottom[0]->width();
		num_ = bottom[0]->num();
		ch_ = bottom[0]->channels();
		ngroup_ = ch_ / group_in_;
		vector<int> weight_shape(4);
		weight_shape[0] = nout_;
		weight_shape[1] = ch_;
		weight_shape[2] = kernel_size_;
		weight_shape[3] = kernel_size_;
		weight_.Reshape(nout_,ch_,kernel_size_,kernel_size_);
		vector<int> bias_shape(1, nout_);
		this->blobs_.resize(2);
		this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
		this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
		//this->blobs_[2].reset(new Blob<Dtype>(h_*w_, ch_, kernel_size_, kernel_size_));
		pindex_ = 0;
		weight_init_ = false;
		init_index();
	}
	
	template <typename Dtype>
	void DecoderConv3Layer<Dtype>::init_index() {
		index_.Reshape(1, 2, h_, w_);
		inv_index_.Reshape(1, 1, h_, w_);
		int * idx = index_.mutable_cpu_data();
		int * inv_idx = inv_index_.mutable_cpu_data();
		int pidx = 0;
		int stride = h_*w_;
		plan_idx_.clear();
		pindex_ = 0;
		//LOG(INFO) << "reinit the index_ to 0";
		for (int pn = 0; pn < h_ + w_ - 1; pn++) {
			plan_idx_.push_back(pidx);
			int ph = pn >= w_ ? pn - w_ + 1 : 0;
			for (int j=0; ph < h_; ph++,j++) {
				int pw = pn - ph;
				if (pw < 0) break;
				idx[pidx] = ph;
				idx[pidx + stride] = pw;
				inv_idx[ph*w_ + pw] = j;
				pidx += 1;
			}
		}
		plan_idx_.push_back(pidx);
		/*
		for (int i = 0; i < h_; i++)
			for (int j = 0; j < w_; j++) 
				LOG(INFO) << idx[i*w_ + j] << " " << idx[i*w_ + j + h_*w_];
		for (int i = 0; i < h_; i++)
			for (int j = 0; j < w_; j++)
				LOG(INFO) << inv_idx[i*w_ + j];
		*/
			
	}
	
	template <typename Dtype>
	void DecoderConv3Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		bool modify = false;
		if (h_ != bottom[0]->height() || w_ != bottom[0]->width()) {
			modify = true;
		}
		h_ = bottom[0]->height();
		w_ = bottom[0]->width();
		if (modify) init_index();
		num_ = bottom[0]->num();
		top[0]->Reshape(num_, nout_, h_, w_);
		mod_ = h_ + w_ + ngroup_ - 2;
		int len = h_ > w_ ? w_ : h_;
		tmp_.Reshape(num_*len*(h_+w_), ch_, kernel_size_,kernel_size_);
		res_.Reshape(num_*len, h_+w_, group_out_, 1);
	}
	
	template <typename Dtype>
	void deocder_conv3_data_to_col_cpu(const int size, const Dtype * input, Dtype * output, const int * index, const int index_stride,
		const int * inv_index, const int kernel_size, const int rchannel, const int height, const int width,
		const int lw, const int start_idx, const int start_sum, const int innershape,
		const int num, const int channel) {
		for (int i = 0; i < size;i++) {
			int pw = i % kernel_size;
			int ph = (i / kernel_size) % kernel_size;
			int pc = (i / kernel_size / kernel_size) % rchannel;
			int pn = (i / kernel_size / kernel_size / rchannel) % num;
			int pb = i / kernel_size / kernel_size / rchannel / num;
			int ps = i % innershape;
			int pidx = pb + start_idx;
			int th = index[pidx];
			int tw = index[pidx + index_stride];
			int lidx = inv_index[th*width + tw];
			int out_idx = (((th + tw - start_sum)*lw + lidx)*num + pn)*innershape + ps;
			int bh = th + ph - kernel_size / 2;
			int bw = tw + pw - kernel_size / 2;
			//output[i] = tw;
			if (bh < 0 || bh >= height || bw < 0 || bw >= width)
			{
				output[out_idx] = 0;
				continue;
			}
			int in_idx = ((pn*channel + pc)*height + bh)*width + bw;
			output[out_idx] = input[in_idx];
		}
	}
	template <typename Dtype>
	void deocder_conv3_col_to_data_cpu(const int size, const Dtype * input, Dtype * output, const int * index, const int index_stride,
		const int * inv_index, const int group_out, const int lw, const int start_idx, const int start_sum, const int psum,
		const int height, const int width, const int nout, const int num, const Dtype * bias) {
		for (int i = 0; i < size;i++) {
			int ps = i % group_out;
			int pb = (i / group_out) / num;
			int pn = (i / group_out) % num;
			int pidx = pb + start_idx;
			int th = index[pidx];
			int tw = index[pidx + index_stride];
			int lidx = inv_index[th*width + tw];
			int in_idx = (((th + tw - start_sum)*lw + lidx)*num + pn)*group_out + ps;
			int pc = psum - th - tw;
			int out_idx = ((pn*nout + pc*group_out + ps)*height + th)*width + tw;
			int bias_idx = pc*group_out + ps;
			output[out_idx] = input[in_idx] + bias[bias_idx];
		}
	}
	template <typename Dtype>
	void conv3_constrains_weight_kernel_v5_cpu(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		for (int index = 0; index < nthreads;index++) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tw + th + tc >= tn + sz - 1)
				weight[index] = Dtype(0);
		}
	}
	template <typename Dtype>
	void conv3_constrains_weight_kernel_v6_cpu(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		for (int index = 0; index < nthreads; index++) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tw + th + tc > tn + sz - 1)
				weight[index] = Dtype(0);
		}
	}
	template <typename Dtype>
	void conv3_transpose_kernel_cpu(const int nthreads, const Dtype * const input, Dtype* const output,
		const int inner_shape, const int ngroup) {
		for (int index = 0; index < nthreads;index++) {
			int ts = index % inner_shape;
			int tn = index / inner_shape;
			int tidx = (ngroup - tn - 1)*inner_shape + ts;
			output[tidx] = input[index];
		}
	}
	template <typename Dtype>
	void DecoderConv3Layer<Dtype>::init_weight_cpu() {
		if (constrain_ == 5) {
			conv3_constrains_weight_kernel_v5_cpu<Dtype>(this->blobs_[0]->count(), 
				this->blobs_[0]->mutable_cpu_data(), ch_, kernel_size_, group_in_, group_out_);
		}
		else if (constrain_ == 6) {
			conv3_constrains_weight_kernel_v6_cpu<Dtype> (this->blobs_[0]->count(), 
				this->blobs_[0]->mutable_cpu_data(), ch_, kernel_size_, group_in_, group_out_);

		}
		weight_init_ = true;
		int inner_shape = group_out_*ch_*kernel_size_*kernel_size_;
		conv3_transpose_kernel_cpu<Dtype> (inner_shape*ngroup_, this->blobs_[0]->cpu_data(), 
			weight_.mutable_cpu_data(), inner_shape, ngroup_);
	}

	template <typename Dtype>
	void DecoderConv3Layer<Dtype>::single_forward_cpu(int la, int lb, int psum, const Dtype * bottom, Dtype * top) {
		int min_t = h_ > w_ ? w_ : h_;
		int max_t = h_ > w_ ? h_ : w_;
		int lw = plan_idx_[lb + 1] - plan_idx_[lb];
		if (la <= max_t - 1 && lb >= min_t - 1) {
			lw = min_t;
		}
		else if (la >= max_t) {
			lw = plan_idx_[la + 1] - plan_idx_[la];
		}
		int ch = psum - la + kernel_size_ > ngroup_ ? ngroup_*group_in_ : (psum - la + kernel_size_)*group_in_;
		int inner_shape = ch*kernel_size_*kernel_size_;
		int cnt = num_*(plan_idx_[lb + 1] - plan_idx_[la])*inner_shape;
		caffe_set(num_*lw*(lb - la + 1)*inner_shape, Dtype(0), tmp_.mutable_cpu_data());
		deocder_conv3_data_to_col_cpu<Dtype> (cnt, bottom, tmp_.mutable_cpu_data(), index_.cpu_data(), h_*w_,
			inv_index_.cpu_data(), kernel_size_, ch, h_, w_, lw, plan_idx_[la], la, inner_shape, num_, ch_);
		const Dtype * weight = weight_.cpu_data();
		int weight_shape = group_out_*ch_*kernel_size_*kernel_size_;
		weight = weight + (ngroup_ - psum + la - 1)*weight_shape;
		caffe_cpu_gemm_batch(CblasNoTrans, CblasTrans, lw*num_, group_out_, inner_shape, Dtype(1.0),
			tmp_.cpu_data(), weight, Dtype(0), res_.mutable_cpu_data(), lw*num_*inner_shape, weight_shape,
			lw*num_*group_out_, lb - la + 1, inner_shape, ch_*kernel_size_*kernel_size_);
	
		cnt = (plan_idx_[lb + 1] - plan_idx_[la])*group_out_*num_;
		const Dtype * bias = this->blobs_[1]->cpu_data();
		deocder_conv3_col_to_data_cpu<Dtype> (cnt, res_.cpu_data(), top, index_.cpu_data(), h_*w_,
			inv_index_.cpu_data(), group_out_, lw, plan_idx_[la], la, psum, h_, w_, nout_, num_, bias);
	}
	template <typename Dtype>
	void DecoderConv3Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		int plan_sum = pindex_;
		if (!weight_init_)
			init_weight();
		pindex_ = (pindex_ + 1) % mod_;
		int la = plan_sum >= ngroup_ ? plan_sum - ngroup_ + 1 : 0;
		int lb = plan_sum > h_ + w_ - 2 ? h_ + w_ - 2 : plan_sum;
		int one_third = (h_ + w_) / 3;
		int two_third = one_third * 2;

		if (la < one_third) {
			if (lb < one_third) {
				single_forward_cpu(la, lb, plan_sum, bottom_data, top_data);
			}
			else if (lb < two_third) {
				single_forward_cpu(la, one_third, plan_sum, bottom_data, top_data);
				single_forward_cpu(one_third, lb, plan_sum, bottom_data, top_data);
			}
			else {
				single_forward_cpu(la, one_third, plan_sum, bottom_data, top_data);
				single_forward_cpu(one_third, two_third, plan_sum, bottom_data, top_data);
				single_forward_cpu(two_third, lb, plan_sum, bottom_data, top_data);
			}

		}
		else if (la < two_third) {
			if (lb < two_third) {
				single_forward_cpu(la, lb, plan_sum, bottom_data, top_data);
			}
			else {
				single_forward_cpu(la, two_third, plan_sum, bottom_data, top_data);
				single_forward_cpu(two_third, lb, plan_sum, bottom_data, top_data);
			}
		}
		else {
			single_forward_cpu(la, lb, plan_sum, bottom_data, top_data);
		}

	}

	template <typename Dtype>
	void DecoderConv3Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(DecoderConv3Layer);
#endif

	INSTANTIATE_CLASS(DecoderConv3Layer);
	REGISTER_LAYER_CLASS(DecoderConv3);

}  // namespace caffe
