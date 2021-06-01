#include "context_reshape.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void context_reshape_opt::init(){
    init_base();
}

void context_reshape_opt::reshape(int num, int channel, int height, int width){
    if(!reshape_base(num,channel,height,width)) return ;
    cpg_ = channel_ / ngroup_;
}

void context_reshape_opt::reshape_top(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_*height_*width_*ngroup_, cpg_});
    reshape_top_base(options,shapes);
}

void context_reshape_opt::reshape_bottom(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(options,shapes);
}


template <typename scalar_t>
__global__ void context_reshape_forward_kernel(const int nthreads, const scalar_t * const bottom, scalar_t * const top,
    const int inner_size, const int channel, const int cpg) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pn = index / inner_size / channel;
        int pc = (index / inner_size) % channel;
        int ps = index % inner_size;
        int tidx = (pn*inner_size*channel / cpg + pc / cpg*inner_size + ps)*cpg + pc%cpg;
        top[tidx] = bottom[index];
    }
}


std::vector<at::Tensor>  context_reshape_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "context_reshape_forward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    context_reshape_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), height_*width_, channel_, cpg_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void context_reshape_backward_kernel(const int nthreads, scalar_t * const bottom, const scalar_t * const top,
    const int inner_size, const int channel, const int cpg) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pn = index / inner_size / channel;
        int pc = (index / inner_size) % channel;
        int ps = index % inner_size;
        int tidx = (pn*inner_size*channel / cpg + pc / cpg *inner_size + ps)*cpg + pc%cpg;
        bottom[index] = top[tidx];
    }
}

std::vector<at::Tensor>  context_reshape_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "context_reshape_backward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    context_reshape_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), height_*width_, channel_, cpg_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return bottom_diff_;
}