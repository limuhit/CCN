#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/ml_quant_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void ml_quant_cal_weight_kernel(const int nthreads, const Dtype* const weight_b, Dtype * const weight, const int levels) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			if (index%levels == 0)
				weight[index] = weight_b[index];
			else
				weight[index] = exp(weight_b[index]);
		}
	}
	template<typename Dtype>
	void MLQuantLayer<Dtype>::cal_weight_gpu() {
		int num = this->blobs_[0]->count();
		const Dtype * const weight_b = this->blobs_[0]->gpu_data();
		Dtype *  const weight = weight_.mutable_gpu_data();
		ml_quant_cal_weight_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
			(num, weight_b, weight,levels_);
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	__global__ void ml_quant_single_forward_kernel(const int nthreads, const Dtype* const val, int * const quant) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			if (val[index] >= 0)
				quant[index]++;
		}
	}
	template <typename Dtype>
	__global__ void ml_quant_single_gpu_forward_kernel(const int num, const Dtype* const bottom, int * const quant,
		Dtype * const top, const Dtype * const weight, Dtype * const count, const int inner_shape,
		const int channels, const int levels) {
		CUDA_KERNEL_LOOP(i, num)
		{
			int pc = (i / inner_shape) % channels;
			Dtype tmp = bottom[i] - weight[pc*levels];
			if (tmp < 0) {
				quant[i] = 0;
				top[i] = weight[pc*levels];
				atomicAdd((float *)(count + pc*levels), float(1.0));
				count[pc*levels]++;
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
	template <typename Dtype>
	__global__ void ml_quant_neighbor_gpu_forward_kernel(const int num, const Dtype* const bottom, int * const quant,
		Dtype * const top, const Dtype * const weight, const int * const quantf, Dtype * const count,
		const int inner_shape, const int channels, const int width, const int levels) {
		CUDA_KERNEL_LOOP(i, num)
		{
			int pc = (i / inner_shape) % channels;
			int pw = i%width;
			int ph = (i%inner_shape) / width;
			int base;
			if (ph > 0 && pw > 0) {
				base = (quantf[i - width] * (levels + 1) + quantf[i - 1])*channels*levels;
			}
			else if (ph > 0 && pw == 0) {
				base = (quantf[i - width] * (levels + 1) + levels)*channels*levels;;
			}
			else if (ph == 0 && pw > 0) {
				base = (levels * (levels + 1) + quantf[i - 1])*channels*levels;
			}
			else {
				base = (levels * (levels + 1) + levels)*channels*levels;
			}
			Dtype tmp = bottom[i] - weight[base + pc*levels];
			if (tmp < 0) {
				quant[i] = 0;
				top[i] = weight[base + pc*levels];
				//count[base + pc*levels]++;
				atomicAdd((float *)(count+base+pc*levels), float(1.0));
				continue;
			}
			int j = 1;
			for (; j < levels; j++)
			{
				tmp -= weight[base + pc*levels + j];
				if (tmp < 0)
					break;
			}
			if (j == levels) j--;
			if (tmp + tmp + weight[base + pc*levels + j] < 0) {
				tmp = tmp + weight[base + pc*levels + j];
				j--;
			}
			top[i] = bottom[i] - tmp;
			quant[i] = j;
			//count[base + pc*levels + j]++;
			atomicAdd((float *)(count + base + pc*levels+j), float(1.0));
		}
	}
	template <typename Dtype>
	__global__ void ml_quant_gpu_copy(const int nthreads, const int * const quant,
		Dtype * const top)
	{
		CUDA_KERNEL_LOOP(index, nthreads) {
			top[index] = quant[index];
		}
	}
	template <typename Dtype>
	void MLQuantLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * const top_data = top[0]->mutable_gpu_data();
		const Dtype * const bottom_data = bottom[0]->gpu_data();
		Dtype * count_data = this->blobs_[2]->mutable_gpu_data();
		int * const quant = quant_.mutable_gpu_data();
		update_weight();
		cal_weight_gpu();
		const Dtype * weight = weight_.gpu_data();
		int num = bottom[0]->count();
		caffe_gpu_scale(this->blobs_[2]->count(), this->blobs_[1]->cpu_data()[1], count_data, count_data);
		//LOG(INFO) << "method:" << method_;
		switch (method_) {
		case 0:
			ml_quant_single_gpu_forward_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
				(num, bottom_data, quant, top_data, weight, count_data, w_*h_, ch_, levels_);
			break;
		case 1:
			ml_quant_single_gpu_forward_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
				(num, bottom_data, quantf_.mutable_gpu_data(),valf_.mutable_gpu_data(), weight, 
					count_data, w_*h_, ch_, levels_);
			weight = weight + levels_*ch_;
			count_data = count_data + levels_ *ch_;
			ml_quant_neighbor_gpu_forward_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
				(num, bottom_data, quant, top_data, weight, quantf_.gpu_data(), count_data, w_*h_, ch_, w_, levels_);
			break;
		default:
			LOG(INFO) << "No Implements!!!";
		}
		if(top.size()>1)
			ml_quant_gpu_copy << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(num, quant, top[1]->mutable_gpu_data());
	}
	template <typename Dtype>
	__global__ void ml_quant_backward_l1_kernel(const int nthreads, Dtype * const weight) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			if (weight[index] < -0.0000001)
				weight[index] = -1.0;
			else if (weight[index] > 0.0000001)
				weight[index] = 1.0;
		}
	}
	template <typename Dtype>
	__global__ void ml_quant_single_gpu_backward_kernel(const int num, const int * const quant,
		const Dtype * const top_diff, Dtype * const weight_diff, const int inner_shape,
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
	template <typename Dtype>
	__global__ void ml_quant_neighbor_gpu_backward_kernel(const int num, const Dtype* const top_diff, const int * const quant,
		Dtype * const weight_diff, const int * const quantf, const int inner_shape,
		const int channels, const int width, const int levels) {
		CUDA_KERNEL_LOOP(i, num)
		{
			int pc = i % channels;
			int idx = (i / channels) % inner_shape + (i / channels / inner_shape)*channels*inner_shape
				+ pc*inner_shape;
			int pw = (i / channels) % width;
			int ph = ((i / channels) % inner_shape) / width;
			int base;
			if (ph > 0 && pw > 0) {
				base = (quantf[idx - width] * (levels + 1) + quantf[idx - 1])*channels*levels;
			}
			else if (ph > 0 && pw == 0) {
				base = (quantf[idx - width] * (levels + 1) + levels)*channels*levels;;
			}
			else if (ph == 0 && pw > 0) {
				base = (levels * (levels + 1) + quantf[idx - 1])*channels*levels;
			}
			else {
				base = (levels * (levels + 1) + levels)*channels*levels;
			}
			for (int j = 0; j <= quant[idx]; j++)
			{
				atomicAdd((float *)(weight_diff + base + pc*levels + j), float(top_diff[idx]));
				//weight_diff[base + pc*levels + j] += top_diff[i];
			}
		}
	}
	template <typename Dtype>
	__global__ void ml_quant_cal_weight_diff_kernel(const int num, Dtype* const weight, 
		const Dtype * const val, const int levels) {
		CUDA_KERNEL_LOOP(i, num)
		{
				if (i%levels != 0)
					weight[i] = weight[i] * val[i];
		}
	}
	template <typename Dtype>
	void MLQuantLayer<Dtype>::cal_weight_diff_gpu() {
		int num = weight_.count();
		Dtype * a = this->blobs_[0]->mutable_gpu_diff();
		const Dtype * b = weight_.gpu_data();
		ml_quant_cal_weight_diff_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
			(num, a,b, levels_);
	}
	template <typename Dtype>
	__global__ void ml_quant_top_diff_kernel(const int num, const Dtype* const weight,
		const Dtype * top_data, const Dtype * bottom_data,
		const int * const quant, const Dtype * const top_diff, Dtype* const bottom_diff, 
		const Dtype alpha, const int level, const int inner_shape, const int channels) {
		CUDA_KERNEL_LOOP(i, num)
		{
			int tc = (i / inner_shape) % channels;
			Dtype beta = 1.0;
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
	template <typename Dtype>
	void MLQuantLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype * top_diff = diff_.mutable_gpu_diff();
		Dtype * weight_diff = this->blobs_[0]->mutable_gpu_diff();
		int num = bottom[0]->count();
		const Dtype * weight = weight_.gpu_data();
		//caffe_gpu_set<int>(quant_.count(), 0, quant_.mutable_gpu_data());
		const int * const quant = quant_.gpu_data();
		caffe_gpu_memcpy(num*sizeof(Dtype), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
		if(top.size()>1)
			ml_quant_top_diff_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
				(num, weight,top[0]->gpu_data(),bottom[0]->gpu_data(),	quant, top[1]->gpu_diff(),
					bottom[0]->mutable_gpu_diff(), this->blobs_[1]->cpu_data()[0],levels_, w_*h_, ch_);
			//caffe_gpu_axpby(num, this->blobs_[1]->cpu_data()[0], top[1]->gpu_diff(), Dtype(1.0),
			//	bottom[0]->mutable_gpu_diff());
		caffe_gpu_memcpy(num*sizeof(Dtype), top[0]->gpu_data(), top_diff);
		caffe_gpu_axpby(num, Dtype(-1.0), bottom[0]->gpu_data(), Dtype(1.0), top_diff);
		Dtype loss;
		caffe_gpu_dot(num, top_diff, top_diff,&loss);
		this->blobs_[1]->mutable_cpu_data()[4] = loss / num;
		caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
		
		switch (method_) {
		case 0:
			ml_quant_single_gpu_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
				(num, quant, top_diff,	weight_diff, w_*h_, ch_, levels_);
			break;
		case 1:
			caffe_gpu_axpby(num, Dtype(-1.0), bottom[0]->gpu_data(), Dtype(1.0), valf_.mutable_gpu_data());
			ml_quant_single_gpu_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
				(num, quantf_.gpu_data(), valf_.gpu_data(),	weight_diff, w_*h_, ch_, levels_);
			weight_diff = weight_diff + levels_*ch_;
			ml_quant_neighbor_gpu_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
				(num, top_diff, quant, weight_diff,	quantf_.gpu_data(), w_*h_, ch_, w_, levels_);
			break;
		default:
			LOG(INFO) << "No Implements!!!";
		}
		cal_weight_diff_gpu();
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MLQuantLayer);

}  // namespace caffe
