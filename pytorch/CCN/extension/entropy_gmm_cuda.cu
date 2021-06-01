#include "entropy_gmm.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void entropy_gmm_opt::init(){
   init_base();
}

void entropy_gmm_opt::reshape(int num, int ng){
    if((num_ == num) &&  (ng_ == ng)) return ;
    num_ = num;
    ng_ = ng;
    assert((ng_==num_gaussian_)&&"the last dim of the weight should be the same as the number of gaussian distributions");
}


void entropy_gmm_opt::reshape_top(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_});
    reshape_top_base(options,shapes);
}

void entropy_gmm_opt::reshape_bottom(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_, ng_});
    shapes.push_back({num_, ng_});
    shapes.push_back({num_, ng_});
    shapes.push_back({num_, 1});
    reshape_bottom_base(options,shapes);
}


template <typename scalar_t>
__global__ void entropy_gmm_forward_kernel(const int nthreads, const scalar_t* const bottom_weight, const scalar_t* const bottom_delta,
    const scalar_t * const bottom_mean, const scalar_t * const label,scalar_t* const weight_diff, scalar_t* const delta_diff,
    scalar_t * const mean_diff, scalar_t * const label_diff,scalar_t * const loss,  const int ng) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        scalar_t s2 = 1. / sqrt(scalar_t(2.0));
        scalar_t sp2 = 1. / sqrt(2.* acos(-1.0));
        scalar_t sum_p = 0;
        label_diff[index] = 0;
        for (int i = 0; i < ng; i++) {
            scalar_t xa = label[index] - 0.5 - bottom_mean[index*ng+i];
            scalar_t xb = label[index] + 0.5 - bottom_mean[index*ng+i];
            scalar_t id = 1. / bottom_delta[index*ng+i];
            scalar_t fa = 0.5 + 0.5*erf(xa * id *s2);
            scalar_t fb = 0.5 + 0.5*erf(xb * id *s2);
            scalar_t p = fb - fa;
            sum_p = sum_p + bottom_weight[index*ng + i] * p;
            scalar_t ga = sp2 * id*exp(-0.5*xa*xa*id*id);
            scalar_t gb = sp2 * id*exp(-0.5*xb*xb*id*id);
            label_diff[index] +=  (gb - ga)*bottom_weight[index*ng+i];
            delta_diff[index*ng+i] = id*(-xb*gb + xa*ga)*bottom_weight[index*ng+i];
            mean_diff[index*ng+i] = (ga - gb)*bottom_weight[index*ng + i];
            weight_diff[index*ng + i] = p;
        }
        loss[index] = -log(sum_p + 0.0000001);
        scalar_t ip = -1. / (sum_p + 0.0000001);
        label_diff[index] *= ip;
        for (int i = 0; i < ng; i++) {
            delta_diff[index*ng + i] *= ip;
            mean_diff[index*ng + i] *= ip;
            weight_diff[index*ng + i] *= ip;
        }
    }
}


std::vector<at::Tensor>  entropy_gmm_opt::forward_cuda(at::Tensor  weight, at::Tensor delta, at::Tensor mean, at::Tensor label) 
{
    reshape(weight.size(0), weight.size(1));
    reshape_top(weight.options());
    reshape_bottom(weight.options());
	int count = num_;
	AT_DISPATCH_FLOATING_TYPES(
		weight.scalar_type(), "entropy_gmm_forward_cuda", 
			([&] {
                    timer_->start();
                    entropy_gmm_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, weight.data_ptr<scalar_t>(), delta.data_ptr<scalar_t>(), mean.data_ptr<scalar_t>(), label.data_ptr<scalar_t>(),
                         bottom_diff_[0].data_ptr<scalar_t>(), bottom_diff_[1].data_ptr<scalar_t>(), bottom_diff_[2].data_ptr<scalar_t>(), bottom_diff_[3].data_ptr<scalar_t>(),
                         top_data_[0].data_ptr<scalar_t>(), ng_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void entropy_gmm_backward_kernel(const int nthreads, scalar_t* const weight_diff, scalar_t* const delta_diff,
    scalar_t * const mean_diff, scalar_t * const label_diff, const scalar_t * const top_diff, const int ng) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pn = index / ng;
        int pg = index % ng;
        if(pg == 0){
            label_diff[pn] *= top_diff[pn];
        }
        weight_diff[index] *= top_diff[pn];
        delta_diff[index] *= top_diff[pn];
        mean_diff[index] *= top_diff[pn];
    }
}

std::vector<at::Tensor>  entropy_gmm_opt::backward_cuda(at::Tensor  top_diff) 
{
    //reshape_bottom(top_diff.options());
	int count = num_* ng_;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "entropy_gmm_backward_cuda", 
			([&] {
                    timer_->start();
                    entropy_gmm_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                       (count, bottom_diff_[0].data_ptr<scalar_t>(), bottom_diff_[1].data_ptr<scalar_t>(), bottom_diff_[2].data_ptr<scalar_t>(),
                        bottom_diff_[3].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), ng_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return bottom_diff_;
}