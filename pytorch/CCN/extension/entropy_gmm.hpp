#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class entropy_gmm_opt: public base_opt{
	public:
		entropy_gmm_opt(int num_gaussian=3, int ignore_label=-1, int device = 0, bool timeit=false){
			num_gaussian_ = num_gaussian;
			ignore_label_ = ignore_label;
			base_opt_init(device,timeit);
		}
		~entropy_gmm_opt(){}
		void init();
		void reshape(int num, int ng);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		int ng_;
		std::vector<at::Tensor>  forward_cuda(at::Tensor  weight, at::Tensor delta, at::Tensor mean, at::Tensor label);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		int num_gaussian_;
		int ignore_label_;
};
