#include "dtow.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void dtow_opt::init(){
   init_base();
}

void dtow_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    if(d2w_){
        h_out_ = height_ * stride_;
        w_out_ = width_ * stride_;
        ch_out_ = channel_ / stride_ / stride_;
    }else{
        h_out_ = height_ / stride_;
        w_out_ = width_ / stride_;
        ch_out_ = channel_ * stride_ * stride_;
    }
}

void dtow_opt::reshape_top(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,ch_out_, h_out_, w_out_});
    reshape_top_base(options,shapes);
}

void dtow_opt::reshape_bottom(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(options,shapes);
}


template <typename scalar_t>
__global__ void dtow_forward_kernel(const int nthreads, const scalar_t* const bottom_data,
    const int num, const int channels, const int height, const int width,
    const int channels_out, const int height_out, const int width_out, const int patch_size,
    scalar_t* const top_data) {
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

template <typename scalar_t>
__global__ void wtod_forward_kernel(const int nthreads, const scalar_t* const bottom_data,
    const int num, const int channels, const int height, const int width,
    const int channels_out, const int height_out, const int width_out, const int patch_size,
    scalar_t* const top_data) {
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


std::vector<at::Tensor>  dtow_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top({bottom_data.options()});
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "dtow_forward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    if(d2w_){
                        dtow_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0 ,stream_ >> >
                            (count, bottom_data.data_ptr<scalar_t>(), num_, channel_, height_, width_, 
                                ch_out_, h_out_, w_out_, stride_, top_data_[0].data_ptr<scalar_t>());
                    }else{
                        wtod_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0 ,stream_ >> >
                            (count, bottom_data.data_ptr<scalar_t>(), num_, channel_, height_, width_, 
                                ch_out_, h_out_, w_out_, stride_, top_data_[0].data_ptr<scalar_t>());
                    }
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void dtow_backward_kernel(const int nthreads, const scalar_t* const top_diff,
    const int num, const int channels, const int height, const int width,
    const int channels_out, const int height_out, const int width_out, const int patch_size,
    scalar_t* const bottom_diff) {
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

template <typename scalar_t>
__global__ void wtod_backward_kernel(const int nthreads, const scalar_t* const top_diff,
    const int num, const int channels, const int height, const int width,
    const int channels_out, const int height_out, const int width_out, const int patch_size,
    scalar_t* const bottom_diff) {
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

std::vector<at::Tensor>  dtow_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom({top_diff.options()});
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "dtow_backward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    if(d2w_){
                        dtow_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0 ,stream_ >> >
                            (count, top_diff.data_ptr<scalar_t>(), num_, channel_, height_, width_, 
                                ch_out_, h_out_, w_out_, stride_, bottom_diff_[0].data_ptr<scalar_t>());
                    }else{
                        wtod_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0 ,stream_ >> >
                            (count, top_diff.data_ptr<scalar_t>(), num_, channel_, height_, width_, 
                                ch_out_, h_out_, w_out_, stride_, bottom_diff_[0].data_ptr<scalar_t>());
                    }
                    
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return bottom_diff_;
}