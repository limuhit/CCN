#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class d_output_opt: public base_opt{
	public:
		d_output_opt(int ngroup, float total_region, int device = 0, bool timeit=false){
			ngroup_ = ngroup;
			total_region_ = total_region;
			base_opt_init(device,timeit);
		}
		~d_output_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		int psum_, nchannel_, mod_, ngroup_;
		float total_region_;
		at::Tensor pindex_;
		std::vector<int> start_idx_;
		at::Tensor tmp_, csum_, cmax_, top_num_;
};
