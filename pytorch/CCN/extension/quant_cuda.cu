#include "quant.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void quant_opt::init(){
    init_base();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_).requires_grad(false);
    weight_ = at::zeros({channel_, bin_num_},options);
    count_data_ = at::zeros({channel_, bin_num_},options);
    iter_ = 0;
}
void quant_opt::reshape(int num, int channel, int height, int width){
    if(!reshape_base(num,channel,height,width)) return ;
    auto options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device_).requires_grad(false);
    quant_ =  at::zeros({num_, channel_, height_, width_},options); 
}

void quant_opt::reshape_top(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    if (ntop_> 1) shapes.push_back({num_,channel_,height_,width_});
    reshape_top_base(options,shapes);
}

void quant_opt::reshape_bottom(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    shapes.push_back({channel_, bin_num_});
    reshape_bottom_base(options,shapes);
}

template <typename scalar_t>
__global__ void ml_quant_cal_weight_kernel(const int nthreads, const scalar_t* const weight_b, scalar_t * const weight, const int levels) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        if (index%levels == 0)
            weight[index] = weight_b[index];
        else
            weight[index] = exp(weight_b[index]);
    }
}


template <typename scalar_t>
__global__ void ml_quant_single_gpu_forward_kernel(const int num, const scalar_t* const bottom, int * const quant,
    scalar_t * const top, const scalar_t * const weight, scalar_t * const count, const int inner_shape,
    const int channels, const int levels) {
    CUDA_KERNEL_LOOP(i, num)
    {
        int pc = (i / inner_shape) % channels;
        scalar_t tmp = bottom[i] - weight[pc*levels];
        if (tmp < 0) {
            quant[i] = 0;
            top[i] = weight[pc*levels];
            atomicAdd((float *)(count + pc*levels), float(1.0));
            //count[pc*levels]++;
            continue;
        }
        int j = 1;
        for (; j < levels; j++)
        {
            tmp -= weight[pc*levels + j];
            if (tmp < 0)
                break;
        }
        if (j == levels) j--;
        if (tmp + tmp + weight[pc*levels + j] < 0) {
            tmp = tmp + weight[pc*levels + j];
            j--;
        }
        top[i] = bottom[i] - tmp;
        quant[i] = j;
        atomicAdd((float *)(count + pc*levels+j), float(1.0));
    }
}

template <typename scalar_t>
__global__ void ml_quant_gpu_copy(const int nthreads, const int * const quant,
    scalar_t * const top)
{
    CUDA_KERNEL_LOOP(index, nthreads) {
        top[index] = quant[index];
    }
}

template <typename scalar_t>
__global__ void ml_quant_check_weight(const int nthreads,  scalar_t * const weight,const scalar_t * const count, const int levels ){
    CUDA_KERNEL_LOOP(i, nthreads) {
        int j = levels - 1;
        for( ; j>1;j--){
            if(count[i*levels + j] >= 1)
                break;
        }
        scalar_t tmp = weight[i*levels+j]-log(static_cast<scalar_t>(levels - j));
        for (; j < levels; j++)
            weight[i*levels + j] = tmp;
        if (count[i*levels] < 1)
        {
            weight[i*levels] = weight[i*levels] + exp(weight[i*levels + 1]);
            tmp = log((exp(weight[i*levels + 1]) + exp(weight[i*levels + 2])) / 2);
            weight[i*levels + 1] = tmp;
            weight[i*levels + 2] = tmp;
            //LOG(INFO) << "update channel " << i;
        }
    }
}

template <typename scalar_t>
__global__ void ml_quant_scale(const int count, scalar_t * input, scalar_t alpha){
    CUDA_KERNEL_LOOP(index, count) {
        input[index] = input[index]*alpha;
    }
}


void quant_opt::update_weight(at::Tensor weight){    
    if (iter_ % mod_ != 0 || iter_ == 0 )  return;
    //printf("check_weights %f\n", weight_decay_);
    AT_DISPATCH_FLOATING_TYPES(
		weight.scalar_type(), "quant_update_weight_cuda", 
			([&] {
                //caffe_gpu_scal(handle_,channel_*bin_num_, static_cast<scalar_t>(weight_decay_), count_data_.data_ptr<scalar_t>());
                
                ml_quant_check_weight<< <CAFFE_GET_BLOCKS(channel_), CAFFE_CUDA_NUM_THREADS, 0 , stream_ >> >
                    (channel_,weight.data_ptr<scalar_t>(),count_data_.data_ptr<scalar_t>(),bin_num_);
                ml_quant_scale<< <CAFFE_GET_BLOCKS(channel_*bin_num_), CAFFE_CUDA_NUM_THREADS, 0 , stream_ >> >
                    (channel_*bin_num_, count_data_.data_ptr<scalar_t>(), static_cast<scalar_t>(weight_decay_));
                
                CUDA_POST_KERNEL_CHECK;
                }
            )
    );
}
	
std::vector<at::Tensor>  quant_opt::quant_forward_cuda(at::Tensor  bottom_data, at::Tensor weight_old, bool train) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
    if(train) update_weight(weight_old);
    //at::Tensor tmp_vec = at::empty({channel_,bin_num_},bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "quant_forward_cuda", 
			([&] {
                    timer_->start();
                    
					count = channel_* bin_num_;
					ml_quant_cal_weight_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0 , stream_ >> >
                        (count, weight_old.data_ptr<scalar_t>(), weight_.data_ptr<scalar_t>(), bin_num_); 
                    count = num_ * channel_ * width_ * height_;
                    ml_quant_single_gpu_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), quant_.data_ptr<int>(), top_data_[0].data_ptr<scalar_t>(), weight_.data_ptr<scalar_t>(), 
                            count_data_.data_ptr<scalar_t>(), width_*height_, channel_, bin_num_);
					if(ntop_>1){
                        ml_quant_gpu_copy << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >(count, quant_.data_ptr<int>(), top_data_[1].data_ptr<scalar_t>());
                    }
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
                    //caffe_gpu_memcpy(bin_num_*channel_*sizeof(scalar_t), count_data_.data_ptr<scalar_t>(),tmp_vec.data_ptr<scalar_t>());
   			    }
			)
    );
    if(train)
        iter_ += 1;
    //printf("iter:%d...\n",iter_);
    return top_data_;
}

template <typename scalar_t>
__global__ void ml_quant_backward_l1_kernel(const int nthreads, scalar_t * const weight) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        if (weight[index] < -0.0000001)
            weight[index] = -1.0;
        else if (weight[index] > 0.0000001)
            weight[index] = 1.0;
    }
}
template <typename scalar_t>
__global__ void ml_quant_single_gpu_backward_kernel(const int num, const int * const quant,
    const scalar_t * const top_diff, scalar_t * const weight_diff, const int inner_shape,
    const int channels, const int levels) {
    CUDA_KERNEL_LOOP(i, num) {
        int pc = i % channels;
        int idx = (i / channels) % inner_shape + (i / channels / inner_shape)*channels*inner_shape
            + pc*inner_shape;
        //int pc = (i / inner_shape) % channels;
        for (int j = 0; j <= quant[idx]; j++)
        {
            atomicAdd((float *)(weight_diff + pc*levels + j), float(top_diff[idx]));
            //weight_diff[pc*levels + j] += top_diff[i];
        }
    }
}
template <typename scalar_t>
__global__ void ml_quant_cal_weight_diff_kernel(const int num, scalar_t* const weight, 
    const scalar_t * const val, const int levels) {
    CUDA_KERNEL_LOOP(i, num)
    {
            if (i%levels != 0)
                weight[i] = weight[i] * val[i];
    }
}

template <typename scalar_t>
__global__ void ml_quant_top_diff_kernel(const int num, const scalar_t* const weight,
    const scalar_t * top_data, const scalar_t * bottom_data,
    const int * const quant, const scalar_t * const top_diff, scalar_t* const bottom_diff, 
    const scalar_t alpha, const int level, const int inner_shape, const int channels) {
    CUDA_KERNEL_LOOP(i, num)
    {
        int tc = (i / inner_shape) % channels;
        scalar_t beta = 1.0;
        if (top_data[i] < bottom_data[i]) {
            beta = quant[i]<level - 1? weight[tc*level + quant[i] + 1]: 10000;
        }
        else if (top_data[i] > bottom_data[i]) {
            beta = quant[i]>0 ? weight[tc*level + quant[i]] : 10000;
        }
        else {
            if (quant[i] == 0) {
                beta = weight[tc*level + quant[i] + 1];
            }
            else if (quant[i] < level - 1) {
                beta = (weight[tc*level + quant[i]] + weight[tc*level + quant[i] + 1]) / 2.0;
            }
            else {
                beta = weight[tc*level + quant[i]];
            }
        }
        if (beta < 0.001) beta = 0.001;
        bottom_diff[i] = bottom_diff[i] + alpha*top_diff[i] / beta;
        
    }
}
std::vector<at::Tensor>  quant_opt::quant_backward_cuda(std::vector<at::Tensor>  top_diff,  at::Tensor bottom_data, at::Tensor top_data){
    
    reshape_bottom(top_diff[0].options());
    int num_thr = num_*channel_*height_*width_;
    AT_DISPATCH_FLOATING_TYPES(
		top_diff[0].scalar_type(), "quant_backward_cuda", 
			([&] {
                    bottom_diff_[0] = top_data - bottom_data;
                    cudaMemset(bottom_diff_[1].data_ptr<scalar_t>(), scalar_t(0.0),  channel_* bin_num_*sizeof(scalar_t));
                    ml_quant_single_gpu_backward_kernel<scalar_t> << <CAFFE_GET_BLOCKS(num_thr), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (num_thr, quant_.data_ptr<int>(), bottom_diff_[0].data_ptr<scalar_t>(),	
                            bottom_diff_[1].data_ptr<scalar_t>(), width_*height_, channel_, bin_num_);
                    ml_quant_cal_weight_diff_kernel<scalar_t> << <CAFFE_GET_BLOCKS(channel_*bin_num_), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (channel_*bin_num_, bottom_diff_[1].data_ptr<scalar_t>(),weight_.data_ptr<scalar_t>(), bin_num_);
                    
                    bottom_diff_[0].copy_(top_diff[0]);
                    
                    if(ntop_>1){
                        ml_quant_top_diff_kernel << <CAFFE_GET_BLOCKS(num_thr), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (num_thr, weight_.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(),bottom_data.data_ptr<scalar_t>(),	
                            quant_.data_ptr<int>(), top_diff[1].data_ptr<scalar_t>(),
                            bottom_diff_[0].data_ptr<scalar_t>(), static_cast<scalar_t>(top_alpha_), bin_num_, width_*height_, channel_);
                    }
                    CUDA_POST_KERNEL_CHECK;
                    
   			    }
			)
    );
    
    return bottom_diff_;
}