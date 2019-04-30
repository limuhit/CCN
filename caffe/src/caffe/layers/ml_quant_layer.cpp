#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/ml_quant_layer.hpp"
namespace caffe {
	template<typename Dtype> 
	void ml_param_init(int num, Dtype *param, Dtype val,int groups) {
		Dtype tmp = log(val);
		for (int i = 0; i < num; i++) {
			if (i%groups == 0)
				param[i] = val / 2;
			else
				param[i] = tmp;
		}
	}
	template<typename Dtype> 
	void MLQuantLayer<Dtype>::cal_weight_cpu() {
		int num=this->blobs_[0]->count();
		const Dtype * const weight_b = this->blobs_[0]->cpu_data();
		Dtype *  const weight = weight_.mutable_cpu_data();
		for (int i = 0; i < num; i++) {
			if (i%levels_ == 0)
				weight[i] = weight_b[i];
			else
				weight[i] = exp(weight_b[i]);
		}
	}
	template <typename Dtype>
	void MLQuantLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		MLQuantParameter rm = this->layer_param_.mlquant_param();
		levels_ = rm.groups();
		this->blobs_.resize(3);
		ch_ = bottom[0]->channels();
		switch (rm.method()) {
			case MLQuantParameter_MLQuantMethod_SINGLE:
				method_ = 0;
				this->blobs_[0].reset(new Blob<Dtype>(1, ch_, 1, levels_));
				this->blobs_[2].reset(new Blob<Dtype>(1, ch_, 1, levels_));
				caffe_set(ch_*levels_, Dtype(0), this->blobs_[2]->mutable_cpu_data());
				ml_param_init(levels_*ch_,this->blobs_[0]->mutable_cpu_data(),Dtype(1.0/levels_),levels_);
				break;
			case MLQuantParameter_MLQuantMethod_NEIGHBOR:
				method_ = 1;
				this->blobs_[0].reset(new Blob<Dtype>((levels_+1)*(levels_+1)+1, ch_, 1, levels_));
				this->blobs_[2].reset(new Blob<Dtype>((levels_ + 1)*(levels_ + 1) + 1, ch_, 1, levels_));
				caffe_set((levels_ + 1)*(levels_ + 1)*ch_*levels_ + ch_*levels_, Dtype(0), this->blobs_[2]->mutable_cpu_data());
				ml_param_init((levels_ + 1)*(levels_ + 1)*ch_*levels_ + ch_*levels_, this->blobs_[0]->mutable_cpu_data(), Dtype(1.0 / levels_), levels_);
				break;
			default:
				LOG(INFO) << "No implement!!!";
				break;
		}
		this->blobs_[1].reset(new Blob<Dtype>(5, 1, 1, 1));
		this->blobs_[1]->mutable_cpu_data()[0] = 1.0;
		this->blobs_[1]->mutable_cpu_data()[1] = 0.999;
		this->blobs_[1]->mutable_cpu_data()[2] = 0;
		this->blobs_[1]->mutable_cpu_data()[3] = 200;
		//this->blobs_[1]->mutable_cpu_data()[5] = 0;
		//this->blobs_[3].reset(new Blob<Dtype>(1, 1, 1, 1));
		//this->blobs_[3]->mutable_cpu_data()[0] = 0;
	}
	template <typename Dtype>
	void MLQuantLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		top[0]->ReshapeLike(*bottom[0]);
		if (top.size()>1)
			top[1]->ReshapeLike(*bottom[0]);
		weight_.ReshapeLike(*this->blobs_[0]);
		diff_.ReshapeLike(*top[0]);
		h_ = bottom[0]->height();
		w_ = bottom[0]->width();
		ch_ = bottom[0]->channels();
		num_ = bottom[0]->num();
		quant_.Reshape(num_,ch_,w_,h_);
		if (method_ > 0)
		{
			valf_.ReshapeLike(*bottom[0]);
			quantf_.ReshapeLike(quant_);
			//this->blobs_[3].reset(new Blob<Dtype>(num_, ch_, w_, h_));
		}
	}
	template <typename Dtype>
	void ml_quant_single_cpu_forward_kernel(const int num, const Dtype* const bottom,int * const quant,
		Dtype * const top, const Dtype * const weight, Dtype * const count,const int inner_shape, 
		const int channels, const int levels) {
		for (int i = 0; i < num; i++)
		{
			int pc = (i / inner_shape)%channels;
			Dtype tmp = bottom[i]-weight[pc*levels];
			if (tmp < 0) {
				quant[i] = 0;
				top[i] = weight[pc*levels];
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
			if (tmp + tmp + weight[pc*levels + j] < 0){
				tmp = tmp + weight[pc*levels + j];
				j--;
			}
			top[i] = bottom[i] - tmp;
			quant[i] = j;
			count[pc*levels+j]++;
		}
	}
	template <typename Dtype>
	void ml_quant_neighbor_cpu_forward_kernel(const int num, const Dtype* const bottom, int * const quant,
		Dtype * const top, const Dtype * const weight, const int * const quantf, Dtype * const count,
		const int inner_shape,const int channels, const int width, const int levels) {
		for (int i = 0; i < num; i++)
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
			Dtype tmp = bottom[i] - weight[base+pc*levels];
			if (tmp < 0) {
				quant[i] = 0;
				top[i] = weight[base+pc*levels];
				count[base + pc*levels]++;
				continue;
			}
			int j = 1;
			for (; j < levels; j++)
			{
				tmp -= weight[base+pc*levels + j];
				if (tmp < 0)
					break;
			}
			if (j == levels) j--;
			if (tmp + tmp + weight[base+pc*levels + j] < 0) {
				tmp = tmp + weight[base+pc*levels + j];
				j--;
			}
			top[i] = bottom[i] - tmp;
			quant[i] = j;
			count[base + pc*levels + j]++;
		}
	}
	template <typename Dtype>
	void ml_quant_cpu_copy(const int num, const int * const quant, Dtype * const top)
	{
		for (int i = 0; i < num; i++)
			top[i] = quant[i];
	}
	template <typename Dtype>
	void ml_quant_check_weight(const int num, Dtype * const weight,const Dtype * const count, const int levels ) {
		for (int i = 0; i < num / levels; i++)
		{
			int j = levels - 1;
			for (; j > 1; j--)
			{
				if (count[i*levels + j] > 0)
					break;
			}
			Dtype tmp = weight[i*levels+j]-log(levels - j);
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
	template <typename Dtype>
	void MLQuantLayer<Dtype>::update_weight() {
		int iter = int(this->blobs_[1]->cpu_data()[2]);
		int model = int(this->blobs_[1]->cpu_data()[3]);
		this->blobs_[1]->mutable_cpu_data()[2] += 1;
		if (iter%model != 0 || iter == 0)
			return;
		ml_quant_check_weight(this->blobs_[0]->count(), this->blobs_[0]->mutable_cpu_data(),
			this->blobs_[2]->cpu_data(), levels_);
		return ;
	}
	template <typename Dtype>
	void MLQuantLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * const top_data = top[0]->mutable_cpu_data();
		const Dtype * const bottom_data = bottom[0]->cpu_data();
		Dtype * count_data = this->blobs_[2]->mutable_cpu_data();
		int * const quant = quant_.mutable_cpu_data();
		update_weight();
		cal_weight_cpu();
		const Dtype * weight = weight_.cpu_data();
		int num = bottom[0]->count();
		caffe_cpu_scale(this->blobs_[2]->count(), this->blobs_[1]->cpu_data()[1], count_data, count_data);
		//LOG(INFO) << "method:" << method_;
		switch (method_) {
			case 0:
				ml_quant_single_cpu_forward_kernel<Dtype>(num, bottom_data, quant, top_data,
					weight,count_data, w_*h_, ch_, levels_);
				break;
			case 1:
				ml_quant_single_cpu_forward_kernel<Dtype>(num, bottom_data, quantf_.mutable_cpu_data(), 
					valf_.mutable_cpu_data(),weight, count_data,w_*h_, ch_, levels_);
				//caffe_copy(num, valf_.cpu_data(), this->blobs_[3]->mutable_cpu_data());
				//LOG(INFO) << "copy complete";
				weight = weight + levels_*ch_;
				count_data = count_data + levels_ *ch_;
				ml_quant_neighbor_cpu_forward_kernel<Dtype>(num, bottom_data, quant, top_data,
					weight, quantf_.cpu_data(), count_data, w_*h_, ch_, w_, levels_);
				break;
			default:
				LOG(INFO) << "No Implements!!!";
		}
		if (top.size() > 1)
			ml_quant_cpu_copy(num, quant, top[1]->mutable_cpu_data());
	}
	template <typename Dtype>
	void ml_quant_single_cpu_backward_kernel(const int num, const int * const quant,
		const Dtype * const top_diff,  Dtype * const weight_diff, const int inner_shape,
		const int channels, const int levels) {
		for (int i = 0; i < num; i++)
		{
			int pc = (i / inner_shape) % channels;
			for (int j = 0; j <= quant[i]; j++)
			{
				weight_diff[pc*levels + j] += top_diff[i];
			}
		}
	}
	template <typename Dtype>
	void ml_quant_neighbor_cpu_backward_kernel(const int num, const Dtype* const top_diff, const int * const quant,
		Dtype * const weight_diff, const int * const quantf, const int inner_shape,
		const int channels, const int width, const int levels) {
		for (int i = 0; i < num; i++)
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
			for (int j = 0; j <= quant[i]; j++)
			{
				weight_diff[base+pc*levels + j] += top_diff[i];
			}
		}
	}
	
	template <typename Dtype>
	void ml_quant_cpu_backward_l1_kernel(const int num, Dtype * const weight) {
		for (int i = 0; i < num; i++)
		{
			if (weight[i] < -0.0000001)
				weight[i] = -1.0;
			else if (weight[i] > 0.0000001)
				weight[i] = 1.0;
		}
	}
	template <typename Dtype>
	void MLQuantLayer<Dtype>::cal_weight_diff_cpu() {
		int num = weight_.count();
		Dtype * a = this->blobs_[0]->mutable_cpu_diff();
		const Dtype * b = weight_.cpu_data();
		for (int i = 0; i < num; i++)
			if (i%levels_ != 0)
				a[i] = a[i] * b[i];

	}
	template <typename Dtype>
	void MLQuantLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype * top_diff = diff_.mutable_cpu_diff();
		Dtype * weight_diff = this->blobs_[0]->mutable_cpu_diff();
		const int * const quant = quant_.cpu_data();
		int num = bottom[0]->count();
		caffe_copy(num, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
		if (top.size()>1)
			caffe_cpu_axpby(num, this->blobs_[1]->cpu_data()[0], top[1]->cpu_diff(),
				Dtype(1.0), bottom[0]->mutable_cpu_diff());
		caffe_copy(num, top[0]->cpu_data(), top_diff);
		caffe_cpu_axpby(num, Dtype(-1.0), bottom[0]->cpu_data(), Dtype(1.0), top_diff);
		this->blobs_[1]->mutable_cpu_data()[4]=caffe_cpu_dot(num, top_diff, top_diff)/num;
		caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
		switch (method_) {
		case 0:
			ml_quant_single_cpu_backward_kernel<Dtype>(num, quant, top_diff,
				weight_diff, w_*h_, ch_, levels_);
			break;
		case 1:
			caffe_cpu_axpby(num, Dtype(-1.0), bottom[0]->cpu_data(), Dtype(1.0), valf_.mutable_cpu_data());
			ml_quant_single_cpu_backward_kernel<Dtype>(num, quantf_.cpu_data(), valf_.cpu_data(),
				weight_diff, w_*h_, ch_, levels_);
			weight_diff = weight_diff + levels_*ch_;
			ml_quant_neighbor_cpu_backward_kernel<Dtype>(num, top_diff, quant, weight_diff,
				quantf_.cpu_data(), w_*h_, ch_, w_, levels_);
			break;
		default:
			LOG(INFO) << "No Implements!!!";
		}
		cal_weight_diff_cpu();

	}

#ifdef CPU_ONLY
	STUB_GPU(MLQuantLayer);
#endif

	INSTANTIATE_CLASS(MLQuantLayer);
	REGISTER_LAYER_CLASS(MLQuant);

}  // namespace caffe
