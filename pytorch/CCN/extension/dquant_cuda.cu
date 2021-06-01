#include "dquant.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void dquant_opt::init(){
    init_base();
}

void dquant_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    weight_ = at::zeros({channel_, bin_num_},at::kFloat).to(torch::Device(torch::kCUDA, device_));
}

void dquant_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_top_base(option,shapes);
}

template <typename scalar_t>
__global__ void decoder_quant_cal_weight_kernel(const int nthreads, const scalar_t* const input, scalar_t * const output, const int level) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        output[index*level] = input[index*level];
        for (int i = 1; i < level; i++) {
            output[index*level + i] = output[index*level+i-1]+exp(input[index*level + i]);
        }
    }
}

template <typename scalar_t>
__global__ void dquant_forward_kernel(const int nthreads, const scalar_t* const input, const scalar_t * const weight,
    scalar_t * const output, const int inner_shape, const int channel, const int level) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tc = (index / inner_shape) % channel;
        int idx = static_cast<int>(input[index]+0.00001);
        output[index] = weight[tc*level + idx];
    }
}

std::vector<at::Tensor>  dquant_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor weight_old) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "dquant_forward_cuda", 
			([&] {
                    timer_->start();
                    count = channel_;
					decoder_quant_cal_weight_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0 , stream_ >> >
                        (count, weight_old.data_ptr<scalar_t>(), weight_.data_ptr<scalar_t>(), bin_num_); 
                    count = num_ * channel_ * width_ * height_;
                    dquant_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, bottom_data.data_ptr<scalar_t>(), weight_.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), width_*height_, channel_, bin_num_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return top_data_;
}


std::vector<at::Tensor>  dquant_opt::backward_cuda(at::Tensor  top_diff) 
{
    
    return {};
}