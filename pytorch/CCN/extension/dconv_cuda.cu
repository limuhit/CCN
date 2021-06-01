#include "dconv.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void dconv_opt::init(){
    init_base();
}

void dconv_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    index_mat_ = at::zeros({height_,width_,2},at::kInt);
    tmp_ = at::zeros({kernel_size_*kernel_size_, group_out_, height*width, num},at::kFloat).to(torch::Device(torch::kCUDA, device_));
	plan_sum_ = 0;
	mod_ = height_ + width_ + ngroup_ - 2;
	int pidx = 0;
	int stride = height_*width_;
	int* idx = index_mat_.data_ptr<int>();
	plan_idx_.clear();
	
	for (int pn = 0; pn < height_ + width_ - 1; pn++) {
		plan_idx_.push_back(pidx);
		int ph = pn >= width_ ? pn - width_ + 1 : 0;
		for (int j=0; ph < height_; ph++,j++) {
			int pw = pn - ph;
			if (pw < 0) break;
			idx[pidx] = ph;
			idx[pidx + stride] = pw;
			pidx += 1;
		}
	}
	plan_idx_.push_back(pidx);
	index_mat_= index_mat_.to(torch::Device(torch::kCUDA, device_));
	//printf("%d %d %d %d\n", height_,width_,mod_,plan_idx_[0]);
}

void dconv_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,nout_,height_,width_});
    reshape_top_base(option,shapes);
}



template <typename scalar_t>
__global__ void deocder_conv_data_to_col_gpu(const int size, const scalar_t * input, const scalar_t* weight, 
    scalar_t * output, const int * index, const int index_stride,const int kernel_size,  
    const int group_in, const int group_out, const int num, const int height, const int width, const int start_idx,
    const int psum, const int inner_shape, const int channel, const int constrain) {
    CUDA_KERNEL_LOOP(i, size) {
        int pn = i % num;
        int pb = (i / num) % inner_shape;
        int pidx = pb + start_idx;
        int th = index[pidx];
        int tw = index[pidx + index_stride];
        int og = (i / num / inner_shape) % group_out;
        int ks = i / num / inner_shape / group_out;
        int kw = ks % kernel_size;
        int kh = ks / kernel_size;
        int half_kernel = kernel_size/2;
        int ph = th - half_kernel + kh;
        int pw = tw - half_kernel + kw;
        if(ph >= height || ph < 0 || pw >= width || pw<0)
            continue;
        scalar_t sum = 0;
        int tc = (psum - th - tw); 
        int nchannel = constrain==5?(psum - ph - pw) * group_in:(psum - ph - pw + 1) * group_in;
        if(nchannel>channel)
            nchannel = channel;
        int skernel = kernel_size * kernel_size;
        int weight_base = (tc * group_out + og)* channel * skernel+ ks;
        int data_base = (pn * channel* height + ph) * width + pw;
        for(int ti = 0; ti < nchannel; ti++){
            sum = sum + input[data_base+ti*index_stride]*weight[weight_base+ti*skernel];
        }
        output[i] = sum;
    }
}
template <typename scalar_t>
__global__ void deocder_conv_col_to_data_gpu(const int size, const scalar_t * input, const scalar_t * bias, scalar_t * output, 
	const int * index, const int index_stride, const int group_out, const int start_idx, const int psum, 
	const int height, const int width, const int nout, const int num, const int inner_shape) {
	CUDA_KERNEL_LOOP(i, size) {
		int pn = i % num;
		int pb = (i / num) % inner_shape;
		int pidx = pb + start_idx;
		int th = index[pidx];
		int tw = index[pidx + index_stride];
		int tc = (psum - th - tw);
		int og = (i / num / inner_shape) % group_out;
		int pout = (tc * group_out + og);
		int out_idx = ((pn*nout+pout)*height+th)*width + tw;
		output[out_idx] = input[i]+bias[pout];
	}
}

template <typename scalar_t>
__global__ void deocder_conv_sum_gpu(const int size, scalar_t * data, const int inner_shape, const int sum_size){
	CUDA_KERNEL_LOOP(i, size) {
		for(int ti = 1; ti< sum_size; ti++)
			data[i] += data[i+ti*inner_shape];	
	}

}



std::vector<at::Tensor>  dconv_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias) 
{
	//printf("here0!\n");
    reshape(bottom_data.size(0), channel_, bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int h_ = height_;
	int w_ = width_;	
	int ch_ = channel_;
	int la = plan_sum_ >= ngroup_ ? plan_sum_ - ngroup_ + 1 : 0;
	int lb = plan_sum_ > h_ + w_ - 2 ? h_ + w_ - 2 : plan_sum_;
	int inner_shape = (plan_idx_[lb + 1] - plan_idx_[la]);
	int skernel = kernel_size_*kernel_size_;
	int cnt = skernel*group_out_*inner_shape*num_;
	//printf("here!\n");
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "dconv_forward_cuda", 
			([&] {
                    timer_->start();
					cudaMemset(tmp_.data_ptr<scalar_t>(), scalar_t(0.0), kernel_size_*kernel_size_* group_out_* inner_shape* num_*sizeof(scalar_t));
					timer_->stop("set zero");
					timer_->start();
					deocder_conv_data_to_col_gpu<scalar_t><<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS, 0 , stream_>>>(
						cnt, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), tmp_.data_ptr<scalar_t>(), index_mat_.data_ptr<int>(),
						h_*w_, kernel_size_, group_in_, group_out_, num_, h_, w_, plan_idx_[la], plan_sum_, inner_shape, ch_, constrain_);
					timer_->stop("kernel 1");
					timer_->start();
					cnt = group_out_*inner_shape*num_;
					deocder_conv_sum_gpu<scalar_t><<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS, 0 , stream_>>>(
						cnt, tmp_.data_ptr<scalar_t>(), cnt, skernel);
					timer_->stop("kernel 2");
					timer_->start();
        			deocder_conv_col_to_data_gpu<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS, 0 , stream_>>>
						(cnt, tmp_.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), 
						 index_mat_.data_ptr<int>(), h_*w_, group_out_, plan_idx_[la], plan_sum_, 
						 h_, w_, nout_, num_, inner_shape);
					timer_->stop("kernel 3");
					CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    plan_sum_ = (plan_sum_ + 1) % mod_;
    return top_data_;
}


std::vector<at::Tensor>  dconv_opt::backward_cuda(at::Tensor  top_diff) 
{

    return {};
}