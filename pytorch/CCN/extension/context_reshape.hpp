#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class context_reshape_opt:public base_opt{
	public:
		context_reshape_opt(int ngroup, int device = 0, bool timeit=false){
			ngroup_ = ngroup;
			base_opt_init(device,timeit);
		}
		~context_reshape_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		int ngroup_, cpg_;
};
