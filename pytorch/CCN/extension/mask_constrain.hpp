#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class mask_constrain_opt:public base_opt{
	public:
		mask_constrain_opt(int constrain=5, int ngroup = 1, int device = 0, bool timeit=false){
			constrain_ = constrain;
			ngroup_ = ngroup;
			base_opt_init(device,timeit);
		}
		~mask_constrain_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
		void  forward_cuda(at::Tensor  bottom_data);
		void  backward_cuda(at::Tensor  top_diff);
		int constrain_, ngroup_, group_in_, group_out_;
};
