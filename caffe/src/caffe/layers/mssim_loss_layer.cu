#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mssim_loss_layer.hpp"
namespace caffe {
	__global__ void sync_ssim_conv_groups() { }
	template <typename Dtype>
	void MssimLossLayer<Dtype>::cuda_conv_backward(Dtype ** bottom, Dtype ** top) {
		const Dtype * weight = weight_.gpu_data();
		for (int i = 0; i < 5; i++) {
			CUDNN_CHECK(cudnnConvolutionBackwardData(
				handle_[i],
				cudnn::dataType<Dtype>::one,
				filter_desc_, weight,
				top_descs_[i], top[i],
				conv_descs_[i],
				bwd_data_algo_[i], workspace[i+5],
				workspace_bwd_data_sizes_[i],
				cudnn::dataType<Dtype>::zero,
				bottom_descs_[i], bottom[i]));
		}
		
		sync_ssim_conv_groups << <1, 1 >> >();
	}
	template <typename Dtype>
	void  MssimLossLayer<Dtype>::cuda_conv_forward(Dtype ** bottom, Dtype ** top) {
		const Dtype * weight = weight_.gpu_data();
		for (int i = 0; i < 5; i++) {
			CUDNN_CHECK(cudnnConvolutionForward(handle_[i],
				cudnn::dataType<Dtype>::one,
				bottom_descs_[i], bottom[i],
				filter_desc_, weight,
				conv_descs_[i],
				fwd_algo_[i], workspace[i], workspace_fwd_sizes_[i],
				cudnn::dataType<Dtype>::zero,
				top_descs_[i], top[i]));
		}
		
		sync_ssim_conv_groups << <1, 1 >> >();
	}
	template <typename Dtype>
	__global__ void caffe_mssim_forward_downsample_gpu_kernel(const int count,  const Dtype * bottom, Dtype * top,
		const int bw, const int bh, const int tw, const int th) {
		CUDA_KERNEL_LOOP(i, count) {
			int pw = i % tw;
			int ph = (i / tw) % th;
			int pn = i / tw / th;
			int bottom_add = (pn*bh + ph*2 )*bw + pw*2;
			Dtype dt = 0;
			if (pw * 2 + 1 < bw) {
				if (ph * 2 + 1 < bh)
					dt = bottom[bottom_add] + bottom[bottom_add + 1] + bottom[bottom_add + bw] + bottom[bottom_add + bw + 1];
				else
					dt = 2 * (bottom[bottom_add] + bottom[bottom_add + 1]);
			}				
			else {
				if (ph * 2 + 1 < bh)
					dt = 2 * (bottom[bottom_add] + bottom[bottom_add + bw]);
				else
					dt = 4 * bottom[bottom_add];
			}
			top[i] = dt / 4;
		}
	}
	template <typename Dtype>
	__global__ void caffe_mssim_forward_gpu_kernel(const int count, const Dtype * x, const Dtype * y,
		const Dtype * xy, const Dtype * x2, const Dtype * y2, Dtype * la, Dtype * lcs, Dtype * ssim,
		const Dtype c1, const Dtype c2) {
		CUDA_KERNEL_LOOP(i, count) {
			Dtype a, b, c, d;
			a = 2 * x[i] * y[i] + c1;
			b = x[i]*x[i] + y[i]*y[i] + c1;
			c = 2 * (xy[i]-x[i]*y[i]) + c2;
			d = x2[i] - x[i]*x[i] + y2[i] - y[i]*y[i] + c2;
			la[i] = a / b;
			lcs[i] = c / d;
			ssim[i] = la[i] * lcs[i];
		}
	}
	template <typename Dtype>
	void MssimLossLayer<Dtype>::multi_sqr(Dtype ** x, Dtype **y) {
		int cnt = 0;
		for (int i = 0; i < 5; i++) {
			cnt = n_*ch_*(out_h_[i])*(out_w_[i]);
			caffe_gpu_sqr(cnt, x[i], y[i]);
		}
	}
	template <typename Dtype>
	void MssimLossLayer<Dtype>::multi_axpby(Dtype ** x, Dtype **y, Dtype alpha) {
		int cnt = 0;
		for (int i = 0; i < 5; i++) {
			cnt = n_*ch_*(out_h_[i])*(out_w_[i]);
			caffe_gpu_axpby(cnt, alpha, x[i], Dtype(1.0), y[i]);
		}
	}
	template <typename Dtype>
	void MssimLossLayer<Dtype>::multi_mul(Dtype ** x, Dtype **y, Dtype **z) {
		int cnt = 0;
		for (int i = 0; i < 5; i++) {
			cnt = n_*ch_*(out_h_[i] )*(out_w_[i] );
			caffe_gpu_mul(cnt, x[i], y[i],z[i]);
		}
	}
	template <typename Dtype>
	void MssimLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * data[5], *label[5],* mx[5],* my[5], *mxy[5],* mx2[5], *my2[5], *tmp[5], *lcs[5], *la[5], *sim[5];
		int cnt;
		for (int i = 0; i < 5; i++) {
			data[i] = data_[i].mutable_gpu_data();
			label[i] = label_[i].mutable_gpu_data();
			mx[i] = x_[i].mutable_gpu_data();
			my[i] = y_[i].mutable_gpu_data();
			mxy[i] = xy_[i].mutable_gpu_data();
			mx2[i] = x2_[i].mutable_gpu_data();
			my2[i] = y2_[i].mutable_gpu_data();
			tmp[i] = tmp_[i].mutable_gpu_data();
			lcs[i] = lcs_[i].mutable_gpu_data();
			la[i] = la_[i].mutable_gpu_data();
			sim[i] = sim_[i].mutable_gpu_data();
		}
		for (int i = 0; i < 5; i++) {
			cnt = n_*ch_*out_h_[i] * out_w_[i];
			if (i == 0) {
				caffe_gpu_memcpy(cnt * sizeof(Dtype), bottom[0]->gpu_data(), data[i]);
				caffe_gpu_memcpy(cnt * sizeof(Dtype), bottom[1]->gpu_data(), label[i]);
			} 
			else {
				caffe_mssim_forward_downsample_gpu_kernel << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
					(cnt, data[i - 1], data[i], out_w_[i - 1], out_h_[i - 1], out_w_[i], out_h_[i]);
				caffe_mssim_forward_downsample_gpu_kernel << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
					(cnt, label[i - 1], label[i], out_w_[i - 1], out_h_[i - 1], out_w_[i], out_h_[i]);
			}
				
		}
		//caffe_gpu_memcpy(data_[3].count() * sizeof(Dtype), data[3], this->blobs_[1]->mutable_gpu_data());
		//caffe_gpu_memcpy(data_[2].count() * sizeof(Dtype), data[2], this->blobs_[0]->mutable_gpu_data());
		//LOG(INFO) << data_[0].count()<<" "<<data_[1].count();
		cuda_conv_forward(data,mx);
		cuda_conv_forward(label, my);
		multi_sqr(data, tmp);
		cuda_conv_forward(tmp, mx2);
		multi_sqr(label, tmp);
		cuda_conv_forward(tmp, my2);
		multi_mul(data, label, tmp);
		cuda_conv_forward(tmp, mxy);
		
		
		for (int i = 0; i < 5; i++) {
			cnt = n_*ch_*(out_w_[i] - 10)*(out_h_[i] - 10);
			caffe_mssim_forward_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
				(cnt, mx[i], my[i], mxy[i], mx2[i], my2[i], la[i], lcs[i], sim[i], c1_, c2_);
		}
		Dtype loss=1,tp; 
		caffe_gpu_set(ones_.count(), Dtype(1), ones_.mutable_gpu_data());
		for (int i = 0; i < 5; i++) {
			cnt = (out_w_[i] - 10) * (out_h_[i] - 10) * ch_*n_;
			if (cnt > 0) {
				if(i<4)	caffe_gpu_dot(cnt, lcs[i], ones_.gpu_data(), &tp);
				else caffe_gpu_dot(cnt, sim[i], ones_.gpu_data(), &tp);
				loss = loss*pow(tp / cnt, gamma_[i]);
				//LOG(INFO) << tp / cnt;
				mcs_[i] = tp / cnt;
			}
		}
		for (int i = 0; i < 5; i++) {
			if (mcs_[i] <= 0) {
				mcs_[i] = 0.0001;
				loss = 0;
			}
		}
		top[0]->mutable_cpu_data()[0] = loss;
	
		
	}
	template <typename Dtype>
	__global__ void caffe_mssim_backward_gpu_kernel(const int count, const Dtype * x, const Dtype * y,
		const Dtype * xy, const Dtype * x2, const Dtype * y2, const Dtype * la, const Dtype * lcs,
		const Dtype c1, const Dtype c2, Dtype * fx, Dtype * fxy, Dtype * fx2 , bool flag, const Dtype alpha) {
		CUDA_KERNEL_LOOP(i, count) {
			Dtype a, b, c, d;
			a = 2 * x[i] * y[i] + c1;
			b = x[i] * x[i] + y[i] * y[i] + c1;
			c = 2 * (xy[i] - x[i] * y[i]) + c2;
			d = x2[i] - x[i] * x[i] + y2[i] - y[i] * y[i] + c2;
			if (flag) {
				fx[i] = ((2 * x[i]*c - 2 * y[i]*d) / d / d) * alpha;
				fx2[i] = (-c / d /d) * alpha;
				fxy[i] = (2 / d) *alpha;
			}
			else {
				fx[i] = (lcs[i] * (2 * y[i]*b - 2 * x[i]*a) / b / b + la[i] * (2 * x[i]*c - 2 * y[i]*d) / d / d)*alpha;
				fx2[i] = (-la[i] * c / d / d)*alpha;
				fxy[i] = (2 * la[i] / d)*alpha;
			}
		}
	}
	template <typename Dtype>
	__global__ void caffe_mssim_backward_downsample_gpu_kernel(const int count, Dtype * bottom, const Dtype * top,
		const int bw, const int bh, const int tw, const int th) {
		CUDA_KERNEL_LOOP(i, count) {
			int pw = i % bw;
			int ph = (i / bw) % bh;
			int pn = i / bw / bh;
			int top_add = (pn*th + ph / 2)*tw + pw / 2;
			bottom[i] = bottom[i] + top[top_add] / 4;
		}
	}
	template <typename Dtype>
	void MssimLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype* const bottom_diff = bottom[0]->mutable_gpu_diff();
		Dtype beta = top[0]->cpu_diff()[0]*top[0]->cpu_data()[0];
		Dtype alpha = Dtype(1.0);
		Dtype * data[5], *label[5], *mx[5], *my[5], *mxy[5], *mx2[5], *my2[5],  *lcs[5], *la[5];
		Dtype * ddata[5], *dmx[5], *dmx2[5],*tmp[5], *dmxy[5];
		int cnt;
		for (int i = 0; i < 5; i++) {
			data[i] = data_[i].mutable_gpu_data();
			label[i] = label_[i].mutable_gpu_data();
			mx[i] = x_[i].mutable_gpu_data();
			my[i] = y_[i].mutable_gpu_data();
			mxy[i] = xy_[i].mutable_gpu_data();
			mx2[i] = x2_[i].mutable_gpu_data();
			my2[i] = y2_[i].mutable_gpu_data();
			lcs[i] = lcs_[i].mutable_gpu_data();
			la[i] = la_[i].mutable_gpu_data();
			ddata[i] = data_[i].mutable_gpu_diff();
			dmx[i] = x_[i].mutable_gpu_diff();
			dmx2[i] = x2_[i].mutable_gpu_diff();
			dmxy[i] = xy_[i].mutable_gpu_diff();
			tmp[i] = tmp_[i].mutable_gpu_data();
		}
		for (int i = 0; i < 5; i++) {
			cnt = n_*ch_*(out_w_[i] - 10)*(out_h_[i] - 10);
			alpha = beta*gamma_[i] / mcs_[i] / cnt;
			if(i<4)
				caffe_mssim_backward_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
					(cnt, mx[i], my[i], mxy[i], mx2[i], my2[i], la[i], lcs[i], c1_, c2_, dmx[i], dmxy[i], dmx2[i], true, alpha);
			else
				caffe_mssim_backward_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
					(cnt, mx[i], my[i], mxy[i], mx2[i], my2[i], la[i], lcs[i], c1_, c2_, dmx[i], dmxy[i], dmx2[i], false, alpha);
		}
		
		cuda_conv_backward(ddata, dmx);
		cuda_conv_backward(tmp, dmxy);
		multi_mul(tmp,label,tmp);
		multi_axpby(tmp, ddata, Dtype(1.0));
		cuda_conv_backward(tmp, dmx2);
		multi_mul(tmp, data, tmp);
		multi_axpby(tmp, ddata, Dtype(2.0));
		for (int i = 4; i >0; i--) {
			cnt = n_*ch_*out_h_[i-1] * out_w_[i-1];
			caffe_mssim_backward_downsample_gpu_kernel << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
					(cnt, ddata[i - 1], ddata[i], out_w_[i - 1], out_h_[i - 1], out_w_[i], out_h_[i]);
		}
		caffe_gpu_memcpy(data_[0].count() * sizeof(Dtype), ddata[0], bottom_diff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MssimLossLayer);

}  // namespace caffe
