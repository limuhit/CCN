#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/entropy_gmm_loss_layer.hpp"
namespace caffe {
	template <typename Dtype>
	__global__ void entropy_gmm_kernel(const int nthreads, const Dtype* const bottom_weight, const Dtype* const bottom_delta,
		const Dtype * const bottom_mean, const Dtype * const label,Dtype* const weight_diff, Dtype* const delta_diff,
		Dtype * const mean_diff, Dtype * const label_diff,Dtype * const loss, int ng) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			Dtype s2 = 1. / sqrt(Dtype(2.0));
			Dtype sp2 = 1. / sqrt(2.* acos(-1.0));
			Dtype sum_p = 0;
			label_diff[index] = 0;
			for (int i = 0; i < ng; i++) {
				Dtype xa = label[index] - 0.5-bottom_mean[index*ng+i];
				Dtype xb = label[index] + 0.5-bottom_mean[index*ng+i];
				Dtype id = 1. / bottom_delta[index*ng+i];
				Dtype fa = 0.5 + 0.5*erf(xa * id *s2);
				Dtype fb = 0.5 + 0.5*erf(xb * id *s2);
				Dtype p = fb - fa;
				sum_p = sum_p + bottom_weight[index*ng + i] * p;
				Dtype ga = sp2 * id*exp(-0.5*xa*xa*id*id);
				Dtype gb = sp2 * id*exp(-0.5*xb*xb*id*id);
				label_diff[index] +=  (gb - ga)*bottom_weight[index*ng+i];
				delta_diff[index*ng+i] = id*(-xb*gb + xa*ga)*bottom_weight[index*ng+i];
				mean_diff[index*ng+i] = (ga - gb)*bottom_weight[index*ng + i];
				weight_diff[index*ng + i] = p;
			}
			loss[index] = -log(sum_p + 0.0000001);
			Dtype ip = -1. / (sum_p + 0.0000001);
			label_diff[index] *= ip;
			for (int i = 0; i < ng; i++) {
				delta_diff[index*ng + i] *= ip;
				mean_diff[index*ng + i] *= ip;
				weight_diff[index*ng + i] *= ip;
			}
		}
	}
	
	template <typename Dtype>
	__global__ void entropy_gmm_check_kernel(const int nthreads, Dtype* const delta) {
		CUDA_KERNEL_LOOP(index, nthreads) {

			if (delta[index] < 0.00001) {
				delta[index] = 0.00001;
			}
		}
	}
	template <typename Dtype>
	void EntropyGmmLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int cnt = bottom[1]->count();
		entropy_gmm_check_kernel<Dtype> << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
			(cnt, bottom[1]->mutable_gpu_data());
		const Dtype* weight = bottom[0]->gpu_data();
		const Dtype* delta = bottom[1]->gpu_data();
		const Dtype* mean = bottom[2]->gpu_data();
		const Dtype* label = bottom[3]->gpu_data();
		Dtype* const weight_diff = bottom[0]->mutable_gpu_diff();
		Dtype* const delta_diff = bottom[1]->mutable_gpu_diff();
		Dtype* const mean_diff = bottom[2]->mutable_gpu_diff();
		Dtype* const label_diff = bottom[3]->mutable_gpu_diff();
		cnt = bottom[3]->count();
		entropy_gmm_kernel<Dtype> << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
				(cnt, weight, delta, mean, label, weight_diff, delta_diff, mean_diff, label_diff,
					diff_.mutable_gpu_data(), label_dim_);
		Dtype loss;
		caffe_gpu_asum(cnt, diff_.gpu_data(), &loss);
		loss = loss / cnt;// / log(2);
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void EntropyGmmLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		//LOG(INFO) << entropy_ << " " << th_<<" "<<wt_;
		int cnt = bottom[3]->count();
		Dtype alpha = top[0]->cpu_diff()[0] / cnt;
		caffe_gpu_scal(cnt*label_dim_, alpha, bottom[0]->mutable_gpu_diff());
		caffe_gpu_scal(cnt*label_dim_, alpha, bottom[1]->mutable_gpu_diff());
		caffe_gpu_scal(cnt*label_dim_, alpha, bottom[2]->mutable_gpu_diff());
		caffe_gpu_scal(cnt*label_dim_, alpha, bottom[3]->mutable_gpu_diff());
	}

	INSTANTIATE_LAYER_GPU_FUNCS(EntropyGmmLossLayer);

}  // namespace caffe
