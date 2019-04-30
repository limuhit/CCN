#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/decoder_output_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void decoder_output_gpu_kernel(const int num, const Dtype * const input, 
		const int * index,  Dtype * const output, const int start_idx, const int len_idx,
		const int height, const int width, const int ngroup, const int nchannel, const int psum) {
		CUDA_KERNEL_LOOP (i,num) {
			int tg = i % ngroup;
			int tl = (i / ngroup) % len_idx;
			int tn = i / ngroup / len_idx;
			int thw = index[tl + start_idx];
			int tw = thw % width;
			int th = thw / width;
			int tc = psum - tw - th;
			int pidx = ((tn*nchannel + tc)*ngroup + tg)*height*width + thw;
			output[i] = input[pidx];
		}

	}
	template <typename Dtype>
	__global__ void decoder_output_table_gpu_kernel(const int num, const Dtype * const input,
		Dtype * const output, const int ngroup, const Dtype base) {
		CUDA_KERNEL_LOOP(index, num) {
			Dtype sum = 0;
			Dtype bias = 0;
			Dtype mval = 0;
			int midx = 0;
			output[index*(ngroup + 1)] = 0;
			for (int i = 0; i < ngroup; i++) {
				sum += input[index*ngroup + i];
				output[index*(ngroup+1) + i+1] = static_cast<int>(sum*base+0.5)+bias;
				
				if (output[index*(ngroup+1) + i +1] == output[index*(ngroup+1) + i])
				{
					bias += 1;
					output[index*(ngroup+1) + i +1] += 1;
				}
				if (output[index*(ngroup+1) + i+1] - output[index*(ngroup+1) + i] > mval) {
						mval = output[index*(ngroup+1) + i + 1] - output[index*(ngroup+1) + i];
						midx = i;
				}
				
			}
		
			if (bias > 0) {
				for (int i = midx; i < ngroup; i++) {
					output[index*(ngroup+1) + i+1] -= bias;
				}
			}
		
			
		}

	}
	template <typename Dtype>
	__global__ void decoder_channel_max(const int num,const Dtype* data, Dtype* out, const int ngroup) {
		CUDA_KERNEL_LOOP(index, num) {
			Dtype maxval = -FLT_MAX;
			for (int c = 0; c < ngroup; ++c) {
				maxval = max(data[index*ngroup+c], maxval);
			}
			out[index] = maxval;
		}
	}

	template <typename Dtype>
	__global__ void decoder_channel_subtract(const int count, Dtype* data, const Dtype* channel_max, const int ngroup) {
		CUDA_KERNEL_LOOP(index, count) {
			int n = index / ngroup;
			data[index] -= channel_max[n];
		}
	}

	template <typename Dtype>
	__global__ void decoder_channel_sum(const int num, const Dtype* data, Dtype* channel_sum, const int ngroup) {
		CUDA_KERNEL_LOOP(index, num) {
			Dtype sum = 0;
			for (int c = 0; c < ngroup; ++c) {
				sum += data[index*ngroup + c];
			}
			channel_sum[index] = sum;
		}
	}

	template <typename Dtype>
	__global__ void decoder_channel_div(const int count, Dtype* data, const Dtype* channel_sum, const int ngroup) {
		CUDA_KERNEL_LOOP(index, count) {
			int n = index / ngroup;
			data[index] /= channel_sum[n];
		}
	}
	template <typename Dtype>
	void DecoderOutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * top_data = top[0]->mutable_gpu_data();
		const Dtype * bottom_data = bottom[0]->gpu_data();
		const int* index = index_.gpu_data();
		int psum = pidx_;
		pidx_ = (pidx_ + 1) % mod_;
		int st = psum - nchannel_ + 1 < 0 ? 0 : psum - nchannel_ + 1;
		int end = psum < height_ + width_ - 2 ? psum + 1 : height_ + width_ - 1;
		int len_idx = start_idx_[end] - start_idx_[st];
		int count = len_idx*num_*ngroup_;
		Dtype * tmp = tmp_.mutable_gpu_data();
		decoder_output_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
			(count, bottom_data, index, tmp, start_idx_[st], len_idx, height_, width_, ngroup_, nchannel_, psum);
		decoder_channel_max<Dtype> << <CAFFE_GET_BLOCKS(len_idx*num_), CAFFE_CUDA_NUM_THREADS >> >
			(len_idx*num_, tmp, cmax_.mutable_gpu_data(), ngroup_);
		decoder_channel_subtract<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, tmp, cmax_.gpu_data(), ngroup_);
		caffe_gpu_exp(count, tmp, tmp);
		decoder_channel_sum<Dtype> << <CAFFE_GET_BLOCKS(len_idx*num_), CAFFE_CUDA_NUM_THREADS >> >
			(len_idx*num_, tmp, csum_.mutable_gpu_data(), ngroup_);
		decoder_channel_div<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, tmp, csum_.gpu_data(), ngroup_);
		//caffe_gpu_memcpy(count * sizeof(Dtype), tmp, top_data);
		decoder_output_table_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(len_idx*num_), CAFFE_CUDA_NUM_THREADS >> >
			(len_idx*num_, tmp, top_data, ngroup_, total_region_);
		top[0]->mutable_cpu_data()[len_idx*num_*(ngroup_ + 1) ] = -1;
	}
	template <typename Dtype>
	void DecoderOutputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

	INSTANTIATE_LAYER_GPU_FUNCS(DecoderOutputLayer);

}  // namespace caffe
