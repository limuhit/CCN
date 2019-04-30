#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mssim_loss_layer.hpp"
namespace caffe {
	template <typename Dtype>
	void MssimLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		//LOG(INFO) << "setup 1";
		w_ = bottom[0]->width();
		h_ = bottom[0]->height();
		ch_ = bottom[0]->channels();
		n_ = bottom[0]->num();
		//this->blobs_.resize(3);
		//this->blobs_[0].reset(new Blob<Dtype>(n_, ch_,h_,w_));
		//this->blobs_[1].reset(new Blob<Dtype>(n_, ch_, h_-10, w_ -10));
		//this->blobs_[2].reset(new Blob<Dtype>(ch_, ch_, 11,11));
		weight_.Reshape(ch_, ch_, 11, 11);
		//kernel_size_ = 11;
		Dtype * weight = weight_.mutable_cpu_data();
		Dtype a= 2.0 * 1.5 * 1.5;
		Dtype sum = 0,tmp=0;
		for (int h = 0; h < 11; h++)
			for (int w = 0; w < 11; w++) {
				tmp = exp(-((h - 5)*(h - 5) + (w - 5)*(w - 5)) / a);
				weight[h * 11 + w] = tmp;
				sum += tmp;
			}
		for (int h = 0; h < 11; h++)
			for (int w = 0; w < 11; w++)
				weight[h * 11 + w] = weight[h * 11 + w] / sum;
		//LOG(INFO) << "setup 2";
		for (int i = 1; i < ch_; i++) {
			for (int h = 0; h < 11; h++)
				for (int w = 0; w < 11; w++)
					weight[i *ch_* 11 * 11+ i*11*11 + h * 11 + w] = weight[h * 11 + w];
		}
		c1_ = (0.01*255)*(0.01*255);
		c2_ = (0.03 * 255)*(0.03 * 255);
		//caffe_copy(ch_*ch_*11*11,weight,this->blobs_[2]->mutable_cpu_data());
		//LOG(INFO) << "setup";
		cuda_setup();
	}
	template<typename Dtype>
	void MssimLossLayer<Dtype>::cuda_setup() {
		//LOG(INFO) << "cudnn_setup";
		workspaceSizeInBytes = 0;
		workspace = new void *[10];
		for (int i = 0; i < 10; i++) workspace[i] = NULL;
		workspaceData = NULL;
		for (int i = 0; i < 5; ++i) {
			fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
			bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
			workspace_fwd_sizes_[i] = 0;
			workspace_bwd_data_sizes_[i] = 0;
			CUDA_CHECK(cudaStreamCreate(&stream_[i]));
			CUDNN_CHECK(cudnnCreate(&handle_[i]));
			CUDNN_CHECK(cudnnSetStream(handle_[i], stream_[i]));
		}
		cudnn::createFilterDesc<Dtype>(&filter_desc_, ch_, ch_, 11, 11);
		for (int i = 0; i < 5; i++) {
			cudnn::createTensor4dDesc<Dtype>(&bottom_descs_[i]);
			cudnn::createTensor4dDesc<Dtype>(&top_descs_[i]);
			cudnn::createConvolutionDesc<Dtype>(&conv_descs_[i]);
		}
		
		handles_setup_ = true;
	}
	template <typename Dtype>
	void MssimLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//LossLayer<Dtype>::Reshape(bottom, top);
		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
		n_ = bottom[0]->num();
		ch_ = bottom[0]->channels();
		//group_ = ch_;
		h_ = bottom[0]->height();
		w_ = bottom[0]->width();
		CHECK_GE(h_, 176) << "the height must be larger than 176";
		CHECK_GE(w_, 176) << "the height must be larger than 176";
		ones_.Reshape(n_, ch_, h_ - 10, w_ - 10);
		int th = 2 * h_, tw = 2 * w_;
		for (int i = 0; i < 5; i++) {
			th = (th + 1) / 2;
			tw = (tw + 1) / 2;
			out_h_[i] = th;
			out_w_[i] = tw;
			//LOG(INFO) << th << " " << tw;
			x_[i].Reshape(n_, ch_,th - 10, tw - 10);
			x2_[i].Reshape(n_, ch_, th - 10, tw - 10);
			xy_[i].Reshape(n_, ch_, th - 10, tw - 10);
			y_[i].Reshape(n_, ch_, th - 10, tw - 10);
			y2_[i].Reshape(n_, ch_,th - 10, tw - 10);
			la_[i].Reshape(n_, ch_, th - 10, tw - 10);
			lcs_[i].Reshape(n_, ch_, th - 10, tw - 10);
			tmp_[i].Reshape(n_, ch_, th, tw);
			sim_[i].Reshape(n_, ch_,th - 10, tw - 10);
			data_[i].Reshape(n_, ch_, th, tw);
			label_[i].Reshape(n_, ch_, th, tw);
		}
		diff_.Reshape(n_,1,1,1);
		cuda_reshape();
	}
	template <typename Dtype>
	void MssimLossLayer<Dtype>::cuda_reshape() {
		//LOG(INFO) << "cuda_reshape";
		size_t workspace_limit_bytes = 8 * 1024 * 1024;
		int pad = 0;
		for (int i = 0; i < 5; i++)
		{
			cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i], n_, ch_, out_h_[i], out_w_[i]);
			cudnn::setTensor4dDesc<Dtype>(&top_descs_[i], n_, ch_, out_h_[i] - 10, out_w_[i] - 10);
			cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i], filter_desc_, pad, pad, 1, 1);
			CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[i],
				filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
				CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
				workspace_limit_bytes, &bwd_data_algo_[i]));
			CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[i],
				filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
				bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]));
			CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[i],
				bottom_descs_[i], filter_desc_, conv_descs_[i], top_descs_[i],
				CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
				workspace_limit_bytes, &fwd_algo_[i]));
			CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[i],
				bottom_descs_[i], filter_desc_, conv_descs_[i], top_descs_[i], fwd_algo_[i],
				&workspace_fwd_sizes_[i]));
		}
		size_t single_space = 0;
		for (int i = 0; i < 5; i++) {
			single_space = workspace_fwd_sizes_[i]>single_space? workspace_fwd_sizes_[i]: single_space;
			single_space = workspace_bwd_data_sizes_[i]> single_space? workspace_bwd_data_sizes_[i]:single_space;
		}
		if (workspaceSizeInBytes < 10*single_space) {
			workspaceSizeInBytes = 10 * single_space;
			cudaFree(this->workspaceData);
			cudaMalloc(&(workspaceData), workspaceSizeInBytes);
			for (int i = 0; i < 10; i++) {
				workspace[i] = reinterpret_cast<char *>(workspaceData)+i*single_space;
			}
		}


	}
	template <typename Dtype>
	void MssimLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	
		
	}

	template <typename Dtype>
	void MssimLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
	}
	template <typename Dtype>
	MssimLossLayer<Dtype>::~MssimLossLayer() {
		for (int i = 0; i < 5; i++) {
			cudnnDestroyTensorDescriptor(bottom_descs_[i]);
			cudnnDestroyTensorDescriptor(top_descs_[i]);
			cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
			cudaStreamDestroy(stream_[i]);
			cudnnDestroy(handle_[i]);
			
		}
		cudnnDestroyFilterDescriptor(filter_desc_); 
		cudaFree(this->workspaceData);

	}
#ifdef CPU_ONLY
	STUB_GPU(MssimLossLayer);
#endif

	INSTANTIATE_CLASS(MssimLossLayer);
	REGISTER_LAYER_CLASS(MssimLoss);

}  // namespace caffe
