#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class dtow_opt: public base_opt{
	public:
		dtow_opt(int stride=2, bool d2w = true, int device=0, bool timeit=false){
			stride_ = stride;
			d2w_ = d2w;
			base_opt_init(device,timeit);
		}
		~dtow_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		int  ch_out_, stride_;
		bool d2w_;
};
