#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class d_extract_opt: public base_opt{
	public:
		d_extract_opt(bool is_label, int device = 0, bool timeit=false){
			is_label_ = is_label;
			base_opt_init(device,timeit);
		}
		~d_extract_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		bool is_label_;
		at::Tensor index_;
		std::vector<int> start_idx_;
		int pidx_, mod_;
};
