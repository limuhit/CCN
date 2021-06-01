#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class dconv_opt: public base_opt{
	public:
		dconv_opt(int channel, int ngroup, int nout, int kernel_size, int constrain, int device = 0, bool timeit=false){
			channel_ = channel;
			ngroup_ = ngroup;
			nout_ = nout;
			kernel_size_ = kernel_size;
			constrain_ = constrain;
			group_in_ = channel / ngroup;
			group_out_ = nout / ngroup;
			base_opt_init(device,timeit);
		}
		~dconv_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		int ngroup_;
		int nout_;
		int kernel_size_;
		int constrain_;
		at::Tensor tmp_, index_mat_;
		std::vector<int> plan_idx_;
		int group_in_, group_out_;
		int plan_sum_, mod_;
};
