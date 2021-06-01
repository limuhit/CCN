#include "d_output.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void d_output_opt::init(){
    init_base();
}

void d_output_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    psum_ = 0;
	mod_ = height + width + ngroup_ - 2;
	nchannel_ = channel / ngroup_;
    pindex_ = at::zeros({height_,width_},at::kInt);
    tmp_ = at::zeros({num_*height_*width_, nchannel_},at::kFloat).to(torch::Device(torch::kCUDA, device_));
	csum_ = at::zeros({num_*height_*width_},at::kFloat).to(torch::Device(torch::kCUDA, device_));
	cmax_ = at::zeros({num_*height_*width_},at::kFloat).to(torch::Device(torch::kCUDA, device_));
    start_idx_.clear();
    int * idx = pindex_.data_ptr<int>();
    start_idx_.clear();
    int index = 0;
    for (int ps = 0; ps < height_+width_-1; ps++) {
        start_idx_.push_back(index);
        for (int i = 0; i < height_; i++) {
            int j = ps - i;
            if (j < 0 || j >= width_)
                continue;
            idx[index] = i*width_ + j;
            index++;
        }
    }
    start_idx_.push_back(index);
    pindex_ = pindex_.to(torch::Device(torch::kCUDA, device_));

}

void d_output_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_*height_*width_,nchannel_+1});
    bool flag = reshape_top_base(option,shapes);
    if (flag){
        top_num_ = at::zeros({1}, option).to(torch::kInt).to(torch::kCPU);
    }
}


template <typename scalar_t>
__global__ void decoder_output_gpu_kernel(const int num, const scalar_t * const input, 
	const int * index,  scalar_t * const output, const int start_idx, const int len_idx,
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

template <typename scalar_t>
__global__ void decoder_output_table_gpu_kernel(const int num, const scalar_t * const input,
	scalar_t * const output, const int ngroup, const scalar_t base) {
	CUDA_KERNEL_LOOP(index, num) {
		scalar_t sum = 0;
		scalar_t bias = 0;
		scalar_t mval = 0;
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
template <typename scalar_t>
__global__ void decoder_channel_max(const int num,const scalar_t* data, scalar_t* out, const int ngroup) {
	CUDA_KERNEL_LOOP(index, num) {
		scalar_t maxval = -FLT_MAX;
		for (int c = 0; c < ngroup; ++c) {
			maxval = max(data[index*ngroup+c], maxval);
		}
		out[index] = maxval;
	}
}

template <typename scalar_t>
__global__ void decoder_channel_subtract(const int count, scalar_t* data, const scalar_t* channel_max, const int ngroup) {
	CUDA_KERNEL_LOOP(index, count) {
		int n = index / ngroup;
		data[index] -= channel_max[n];
	}
}

template <typename scalar_t>
__global__ void decoder_channel_sum(const int num, const scalar_t* data, scalar_t* channel_sum, const int ngroup) {
	CUDA_KERNEL_LOOP(index, num) {
		scalar_t sum = 0;
		for (int c = 0; c < ngroup; ++c) {
			sum += data[index*ngroup + c];
		}
		channel_sum[index] = sum;
	}
}

template <typename scalar_t>
__global__ void decoder_channel_div(const int count, scalar_t* data, const scalar_t* channel_sum, const int ngroup) {
	CUDA_KERNEL_LOOP(index, count) {
		int n = index / ngroup;
		data[index] /= channel_sum[n];
	}
}

template <typename scalar_t>
__global__ void decoder_exp(const int count, const scalar_t * input, scalar_t* output){
	CUDA_KERNEL_LOOP(index, count) {
		output[index] = exp(input[index]);
	}
}

std::vector<at::Tensor>  d_output_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int st = psum_ - ngroup_ + 1 < 0 ? 0 : psum_ - ngroup_ + 1;
	int end = psum_ < height_ + width_ - 2 ? psum_ + 1 : height_ + width_ - 1;
	int len_idx = start_idx_[end] - start_idx_[st];
	int count = len_idx*num_*nchannel_;
	top_num_.data_ptr<int>()[0] = len_idx*num_;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "d_output_forward_cuda", 
			([&] {
                timer_->start();
                scalar_t * tmp = tmp_.data_ptr<scalar_t>();
                decoder_output_gpu_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                    (count, bottom_data.data_ptr<scalar_t>(), pindex_.data_ptr<int>(), tmp, start_idx_[st], len_idx, height_, width_, nchannel_, ngroup_, psum_);
                decoder_channel_max << <CAFFE_GET_BLOCKS(len_idx*num_), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (len_idx*num_, tmp, cmax_.data_ptr<scalar_t>(), nchannel_);
                decoder_channel_subtract << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, tmp, cmax_.data_ptr<scalar_t>(),nchannel_);
                decoder_exp<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, tmp, tmp);
                decoder_channel_sum << <CAFFE_GET_BLOCKS(len_idx*num_), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (len_idx*num_, tmp, csum_.data_ptr<scalar_t>(), nchannel_);
                decoder_channel_div << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, tmp, csum_.data_ptr<scalar_t>(), nchannel_);
                decoder_output_table_gpu_kernel << <CAFFE_GET_BLOCKS(len_idx*num_), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (len_idx*num_, tmp, top_data_[0].data_ptr<scalar_t>(), nchannel_, static_cast<scalar_t>(total_region_));
                CUDA_POST_KERNEL_CHECK;
                timer_->stop("kernel d_output");
   			    }
			)
    );
    psum_ = (psum_ + 1) % mod_;
    return {top_data_[0],top_num_};
}

std::vector<at::Tensor>  d_output_opt::backward_cuda(at::Tensor  top_diff) 
{
    return {};
}