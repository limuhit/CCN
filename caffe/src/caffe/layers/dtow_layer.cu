#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dtow_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void dtow_kernel(const int nthreads, const Dtype* const bottom_data,
		const int num, const int channels, const int height, const int width,
		const int channels_out, const int height_out, const int width_out, const int patch_size,
		Dtype* const top_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index%width;
			int th = (index / width) % height;
			int tc = (index / width / height) % channels;
			int	tn = index / width / height / channels;
			int p2size = patch_size*patch_size;
			int pc = tc / p2size;
			int rc = tc % p2size;
			int ph = th*patch_size + rc / patch_size;
			int pw = tw*patch_size + rc % patch_size;
			int pidx = ((tn*channels_out + pc)*height_out + ph)*width_out + pw;
			top_data[pidx] = bottom_data[index];

		}
	}
	template <typename Dtype>
	__global__ void wtod_kernel(const int nthreads, const Dtype* const bottom_data,
		const int num, const int channels, const int height, const int width,
		const int channels_out, const int height_out, const int width_out, const int patch_size,
		Dtype* const top_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index%width;
			int th = (index / width) % height;
			int tc = (index / width / height) % channels;
			int	tn = index / width / height / channels;
			int p2size = patch_size*patch_size;
			int ph = th / patch_size;
			int pw = tw / patch_size;
			int pc = tc * p2size + (th%patch_size)*patch_size + tw%patch_size;
			int pidx = ((tn*channels_out + pc)*height_out + ph)*width_out + pw;
			top_data[pidx] = bottom_data[index];

		}
	}
	template <typename Dtype>
	__global__ void dtow_zig_gpu_kernel(const int count, const Dtype * const bottom, Dtype * const top,
		const int * index, const int psize, const int width, const int height, const int channel, const int ngroup) {
		CUDA_KERNEL_LOOP(i,count)
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
			top[tidx] = bottom[i];
		}
	}
	template <typename Dtype>
	__global__ void wtod_zig_gpu_kernel(const int count, const Dtype * const bottom, Dtype * const top,
		const int * index, const int psize, const int width, const int height, const int channel, const int ngroup) {
		CUDA_KERNEL_LOOP(i, count)
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
	void DtowLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* const top_data = top[0]->mutable_gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		int count = bottom[0]->count();
		if (d2w){
			if (ld_) {
				//LOG(INFO) << ngroup_;
				dtow_zig_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
					(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data(),
					idx_.cpu_data(), psize, w_in, h_in, ch_in,ngroup_);
			}
			else {
				dtow_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
					(count, bottom_data, num_, ch_in, h_in, w_in, ch_out, h_out, w_out, psize, top_data);
			}
			
		}
		else{
			if (ld_) {
				wtod_zig_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
					(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data(),
					idx_.cpu_data(), psize, w_out, h_out, ch_out,ngroup_);
			}
			else {
				wtod_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
					(count, bottom_data, num_, ch_in, h_in, w_in, ch_out, h_out, w_out, psize, top_data);
			}
			
		}
		
		CUDA_POST_KERNEL_CHECK;
	}
	template <typename Dtype>
	__global__ void dtow_backward_kernel(const int nthreads, const Dtype* const top_diff,
		const int num, const int channels, const int height, const int width,
		const int channels_out, const int height_out, const int width_out, const int patch_size,
		Dtype* const bottom_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index%width;
			int th = (index / width) % height;
			int tc = (index / width / height) % channels;
			int	tn = index / width / height / channels;
			int p2size = patch_size*patch_size;
			int pc = tc / p2size;
			int rc = tc % p2size;
			int ph = th*patch_size + rc / patch_size;
			int pw = tw*patch_size + rc % patch_size;
			int pidx = ((tn*channels_out + pc)*height_out + ph)*width_out + pw;
			bottom_diff[index] = top_diff[pidx];
		}
	}
	template <typename Dtype>
	__global__ void wtod_backward_kernel(const int nthreads, const Dtype* const top_diff,
		const int num, const int channels, const int height, const int width,
		const int channels_out, const int height_out, const int width_out, const int patch_size,
		Dtype* const bottom_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index%width;
			int th = (index / width) % height;
			int tc = (index / width / height) % channels;
			int	tn = index / width / height / channels;
			int p2size = patch_size*patch_size;
			int ph = th / patch_size;
			int pw = tw / patch_size;
			int pc = tc * p2size + (th%patch_size)*patch_size + tw%patch_size;
			int pidx = ((tn*channels_out + pc)*height_out + ph)*width_out + pw;
			bottom_diff[index] = top_diff[pidx];
		}
	}
	template <typename Dtype>
	void DtowLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* const top_diff = top[0]->gpu_diff();
		Dtype* const bottom_diff = bottom[0]->mutable_gpu_diff();
		int count = bottom[0]->count();
		if (d2w){
			if (ld_) {
				wtod_zig_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
					(bottom[0]->count(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(),
					idx_.cpu_data(), psize, w_in, h_in, ch_in,ngroup_);
			}
			else {
				dtow_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
					(count, top_diff, num_, ch_in, h_in, w_in, ch_out, h_out, w_out, psize, bottom_diff);
			}
			
		}
		else{
			if (ld_) {
				dtow_zig_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
					(bottom[0]->count(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(),
					idx_.cpu_data(), psize, w_out, h_out, ch_out,ngroup_);
			}
			else {
				wtod_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
					(count, top_diff, num_, ch_in, h_in, w_in, ch_out, h_out, w_out, psize, bottom_diff);
			}
			
		}
		
		//LOG(INFO) << "1";
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DtowLayer);

}  // namespace caffe
