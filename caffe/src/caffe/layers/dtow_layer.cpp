#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dtow_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void DtowLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		DtowParameter rm = this->layer_param_.dtow_param();
		psize = rm.psize();
		idx_.Reshape(2, 1, 1, psize*psize);
		ngroup_ = rm.ngroup();
		d2w = (rm.method() == DtowParameter_DtowMethod_MDTOW || rm.method() == DtowParameter_DtowMethod_LDTOW);
		ld_ = (rm.method() == DtowParameter_DtowMethod_LWTOD || rm.method() == DtowParameter_DtowMethod_LDTOW);
		if (ld_) {
			int xi = 0;
			int * index = idx_.mutable_cpu_data();
			for (int i = 0; i < 2 * psize - 1; i++) {
				int h_s = i < psize ? 0 : i - psize + 1;
				int h_e = i >= psize ? psize : i + 1;
				for (int j = h_s; j < h_e; j++)
				{
					int pw = xi % psize;
					int ph = xi / psize;
					index[xi] = j-ph;
					index[xi + psize*psize] = i - j - pw;
					xi++;
				}
			}
			/*
			for (int i = 0; i < psize; i++)
				for(int j=0;j<psize;j++)
					LOG(INFO) << i+index[i*psize+j] << " " << j+index[i*psize+ j + psize*psize];
			*/
		}

	}
	template <typename Dtype>
	void DtowLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		h_in = bottom[0]->height();
		w_in = bottom[0]->width();
		ch_in = bottom[0]->channels();
		num_ = bottom[0]->num();
		DtowParameter rm = this->layer_param_.dtow_param();
		
		if (d2w){
			CHECK_EQ(0, ch_in % (psize*psize)) << "the size of depth must be multiple of the square of the upsampling size";
			ch_out = ch_in / (psize*psize);
			h_out = h_in * psize;
			w_out = w_in * psize;
		}
		else{
			CHECK_EQ(0, w_in % psize) << "the size of width must be multiple of the sampling size";
			CHECK_EQ(0, h_in % psize) << "the size of height must be multiple of the sampling size";
			ch_out = ch_in * psize*psize;
			h_out = h_in / psize;
			w_out = w_in / psize;
		}
		top[0]->Reshape(num_, ch_out, h_out, w_out);
	}
	template <typename Dtype>
	void dtow_zig_cpu_kernel(const int count, const Dtype * const bottom, Dtype * const top,
		const int * index, const int psize, const int width, const int height, const int channel, const int ngroup) {
		for (int i = 0; i < count; i++)
		{
			int mod = psize*psize*ngroup;
			int tchannel = channel / mod * ngroup;
			int pw = i % width;
			int ph = (i / width) % height;
			int pc = (i / width / height) % channel;
			int pn = i / width / height / channel;
			int tc = pc / mod * ngroup;
			int tt = pc % mod;
			tc = tc + tt % ngroup;
			tt = tt / ngroup;
			int fh = tt / psize;
			int fw = tt % psize;
			int th = ph*psize + fh + index[fh*psize+fw];
			int tw = pw*psize + fw + index[fh*psize+fw+psize*psize];
			int tidx = ((pn*tchannel + tc)*height*psize + th)*width*psize + tw;
			top[tidx] = bottom[i];
		}
	}
	template <typename Dtype>
	void wtod_zig_cpu_kernel(const int count, const Dtype * const bottom, Dtype * const top,
		const int * index, const int psize, const int width, const int height, const int channel,const int ngroup) {
		for (int i = 0; i < count; i++)
		{
			int mod = psize*psize*ngroup;
			int tchannel = channel / mod * ngroup;
			int pw = i % width;
			int ph = (i / width) % height;
			int pc = (i / width / height) % channel;
			int pn = i / width / height / channel;
			int tc = pc / mod * ngroup;
			int tt = pc % mod;
			tc = tc + tt % ngroup;
			tt = tt / ngroup;
			int fh = tt / psize;
			int fw = tt % psize;
			int th = ph*psize + fh + index[fh*psize + fw];
			int tw = pw*psize + fw + index[fh*psize + fw + psize*psize];
			int tidx = ((pn*tchannel + tc)*height*psize + th)*width*psize + tw;
			top[i] = bottom[tidx];
		}
	}
	template <typename Dtype>
	void DtowLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * const top_data = top[0]->mutable_cpu_data();
		const Dtype * const bottom_data = bottom[0]->cpu_data();
		int p2size = psize*psize;
		const int stride = ch_in*w_in*h_in;
		int pc, rc, ph, pw, pidx;
		int tc, tw, th, tn;
		if (d2w){
			if (ld_) {
				dtow_zig_cpu_kernel<Dtype>(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data(),
					idx_.cpu_data(),psize,w_in,h_in,ch_in,ngroup_);
			}
			else {
				for (int i = 0; i < num_*stride; i++)
				{
					tw = i%w_in;
					th = (i / w_in) % h_in;
					tc = (i / w_in / h_in) % ch_in;
					tn = i / w_in / h_in / ch_in;
					pc = tc / p2size;
					rc = tc % p2size;
					ph = th*psize + rc / psize;
					pw = tw*psize + rc % psize;
					pidx = tn*stride + (pc*h_out + ph)*w_out + pw;
					top_data[pidx] = bottom_data[i];
				}
			}
			
		}
		else{
			if (ld_) {
				wtod_zig_cpu_kernel<Dtype>(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data(),
					idx_.cpu_data(), psize, w_out, h_out, ch_out,ngroup_);
			}
			else {
				for (int i = 0; i < num_*stride; i++)
				{
					tw = i%w_in;
					th = (i / w_in) % h_in;
					tc = (i / w_in / h_in) % ch_in;
					tn = i / w_in / h_in / ch_in;
					ph = th / psize;
					pw = tw / psize;
					pc = tc * p2size + (th%psize)*psize + tw%psize;
					pidx = tn*stride + (pc*h_out + ph)*w_out + pw;
					top_data[pidx] = bottom_data[i];
				}
			}
			
		}
		


	}

	template <typename Dtype>
	void DtowLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype * const top_diff = top[0]->cpu_diff();
		Dtype * const bottom_diff = bottom[0]->mutable_cpu_diff();
		int p2size = psize*psize;
		const int stride = ch_in*w_in*h_in;
		int pc, rc, ph, pw, pidx;
		int tc, tw, th, tn;
		if (d2w){
			if (ld_) {
				wtod_zig_cpu_kernel<Dtype>(bottom[0]->count(), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff(),
					idx_.cpu_data(), psize, w_in, h_in, ch_in,ngroup_);
			}
			else {
				for (int i = 0; i < num_*stride; i++)
				{
					tw = i%w_in;
					th = (i / w_in) % h_in;
					tc = (i / w_in / h_in) % ch_in;
					tn = i / w_in / h_in / ch_in;
					pc = tc / p2size;
					rc = tc % p2size;
					ph = th*psize + rc / psize;
					pw = tw*psize + rc % psize;
					pidx = tn*stride + (pc*h_out + ph)*w_out + pw;
					bottom_diff[i] = top_diff[pidx];
				}
			}
			
		}
		else{
			if (ld_) {
				dtow_zig_cpu_kernel<Dtype>(bottom[0]->count(), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff(),
					idx_.cpu_data(), psize, w_out, h_out, ch_out,ngroup_);
			}
			else {
				for (int i = 0; i < num_*stride; i++)
				{
					tw = i%w_in;
					th = (i / w_in) % h_in;
					tc = (i / w_in / h_in) % ch_in;
					tn = i / w_in / h_in / ch_in;
					ph = th / psize;
					pw = tw / psize;
					pc = tc * p2size + (th%psize)*psize + tw%psize;
					pidx = tn*stride + (pc*h_out + ph)*w_out + pw;
					bottom_diff[i] = top_diff[pidx];
				}
			}
			
		}
		
	}

#ifdef CPU_ONLY
	STUB_GPU(DtowLayer);
#endif

	INSTANTIATE_CLASS(DtowLayer);
	REGISTER_LAYER_CLASS(Dtow);

}  // namespace caffe
