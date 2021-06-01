#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include <TH/THBlas.h>
#include "base_opt.hpp"
class quant_opt:public base_opt{
	public:
		quant_opt(int channel, int bin_num, float weight_decay=0.9, int check_iters=100, int ntop=1, float top_alpha=0.1, int device=0, bool timeit=false){
			channel_ = channel;
			bin_num_ = bin_num;
			mod_ = check_iters;
			ntop_ = ntop;
			weight_decay_ = weight_decay;
			top_alpha_ = top_alpha;
			base_opt_init(device,timeit);
		}
		~quant_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void update_weight(at::Tensor weight);
		void reshape_bottom(at::TensorOptions options);
		at::Tensor count_data_, quant_, weight_;
		std::vector<at::Tensor>  quant_forward_cuda(at::Tensor  bottom_data, at::Tensor weight, bool train);
		std::vector<at::Tensor>  quant_backward_cuda(std::vector<at::Tensor>  top_diff, at::Tensor bottom_data, at::Tensor top_data);
		float weight_decay_, top_alpha_;
		int bin_num_;
		int iter_, mod_, ntop_;
};
