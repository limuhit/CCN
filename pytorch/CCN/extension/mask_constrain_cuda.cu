#include "mask_constrain.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void mask_constrain_opt::init(){
    init_base();
}

void mask_constrain_opt::reshape(int num, int channel, int height, int width){
    if(!reshape_base(num,channel,height,width)) return ;
    group_in_ = channel_ / ngroup_;
    group_out_ = num_ / ngroup_;
}
template <typename scalar_t>
__global__ void conv_mask_v5_kernel(const int nthreads, scalar_t* const weight,
    const int channel, const int sz, const int group_in, const int group_out){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % sz;
        int th = (index / sz) % sz;
        int tc = (index / sz / sz) % channel / group_in;
        int tn = index / sz / sz / channel / group_out;
        if (tw + th + tc >= tn + sz - 1)
            weight[index] = scalar_t(0);
    }
    
}

template <typename scalar_t>
__global__ void conv_mask_v6_kernel(const int nthreads, scalar_t* const weight,
    const int channel, const int sz, const int group_in, const int group_out){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % sz;
        int th = (index / sz) % sz;
        int tc = (index / sz / sz) % channel / group_in;
        int tn = index / sz / sz / channel / group_out;
        if (tw + th + tc > tn + sz - 1)
            weight[index] = scalar_t(0);
    }
}
template <typename scalar_t>
__global__ void mask_constrain_forward_kernel(const int nthreads, const scalar_t* const input,  
     scalar_t * const output, const int inner_shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        output[index] = input[index];
    }
}


void  mask_constrain_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
	int count = num_ * channel_ * width_ * height_;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "mask_constrain_forward_cuda", 
			([&] {
                    timer_->start();
                    if(constrain_==5){
                        conv_mask_v5_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                        (count, bottom_data.data_ptr<scalar_t>(), channel_, width_, group_in_, group_out_);
                    }else{
                        conv_mask_v6_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                        (count, bottom_data.data_ptr<scalar_t>(), channel_, width_, group_in_, group_out_);
                    }
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return ;
}


void   mask_constrain_opt::backward_cuda(at::Tensor  top_diff) 
{
	int count= num_ * channel_ * width_ * height_;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "mask_constrain_backward_cuda", 
			([&] {
                    timer_->start();
                    if(constrain_==5){
                        conv_mask_v5_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                        (count, top_diff.data_ptr<scalar_t>(), channel_, width_, group_in_, group_out_);
                    }else{
                        conv_mask_v6_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                        (count, top_diff.data_ptr<scalar_t>(), channel_, width_, group_in_, group_out_);
                    }
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return ;
}