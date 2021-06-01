#include "d_extract.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"
void d_extract_opt::init(){
    init_base();
}

void d_extract_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    pidx_ = 0;
    mod_ = height_ + width_ + channel_ - 2;
    stream_ = at::cuda::getCurrentCUDAStream();
    index_ = at::zeros({height_,width_},at::kInt);
    int * idx = index_.data_ptr<int>();
    start_idx_.clear();
    int index = 0;
    for (int ps = 0; ps < height_ + width_ - 1; ps++) {
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
    index_ = index_.to(torch::Device(torch::kCUDA, device_));
}

void d_extract_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,1,height_,width_});
    reshape_top_base(option,shapes);
}


template <typename scalar_t>
__global__ void d_extract_forward_kernel(const int num, const scalar_t * const input,
    const int * index, scalar_t * const output, const int start_idx, const int len_idx,
    const int height, const int width, const int nchannel, const int psum) {
    CUDA_KERNEL_LOOP(i, num) {

        int tl = i  % len_idx;
        int tn = i / len_idx;
        int thw = index[tl + start_idx];
        int tw = thw % width;
        int th = thw / width;
        int tc = psum - tw - th;
        int pidx = (tn*nchannel + tc)*height*width + thw;
        output[i] = input[pidx];
    }

}


std::vector<at::Tensor>  d_extract_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	const int* index = index_.data_ptr<int>();
    int psum = pidx_;
    pidx_ = (pidx_ + 1) % mod_;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "d_extract_forward_cuda", 
			([&] {
                const scalar_t * bottom = bottom_data.data_ptr<scalar_t>();
                scalar_t * top_data = top_data_[0].data_ptr<scalar_t>();
                
                if (is_label_) {
                    int st = psum - channel_ + 1 < 0 ? 0 : psum - channel_ + 1;
                    int end = psum < height_ + width_ - 2 ? psum + 1 : height_ + width_ - 1;
                    int len_idx = start_idx_[end] - start_idx_[st];
                    int count = len_idx*num_ * 1;
                    d_extract_forward_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom, index, top_data, start_idx_[st], len_idx, height_, width_, channel_, psum);
                }
                else {
                    if (psum == 0) {
                        caffe_gpu_set(stream_, num_*width_*height_, scalar_t(0), top_data);
                    }
                    else {
                        psum -= 1;
                        int st = psum - channel_ + 1 < 0 ? 0 : psum - channel_ + 1;
                        int end = psum < height_ + width_ - 2 ? psum + 1 : height_ + width_ - 1;
                        int len_idx = start_idx_[end] - start_idx_[st];
                        int count = len_idx*num_ * 1;
                        d_extract_forward_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom, index, top_data, start_idx_[st], len_idx, height_, width_, channel_, psum);
                    }
                }
                
                CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}



std::vector<at::Tensor>  d_extract_opt::backward_cuda(at::Tensor  top_diff) 
{
    return {};
}