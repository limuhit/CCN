#pragma once
#include "ext_all.hpp" 
#include "timer.h"
class base_opt{
	public:
		base_opt(){	}
        void base_opt_init(int device, bool time_it){
            timer_ = new Timer(device, time_it);
            to(device);
        }
		~base_opt(){}
		void to(int device){
			if(device_ == device)
                return;
			device_ = device;
			timer_ ->to(device_);
			init();
		}
        virtual void init() = 0;
		void init_base(){
            at::cuda::set_device(device_);
            auto s = at::cuda::getCurrentCUDAStream();
            stream_ = s.stream();
            init_bottom_ = false;
            init_top_ = false;
            num_ = -1;
        }
		bool reshape_base(int num, int channel, int height, int width){
             if((num_ == num) && (channel_ == channel) && (height_ == height) && (width_ == width)) return false;
                num_ = num;
                channel_ = channel;
                height_ = height;
                width_ = width;
                return true;
        }
        bool is_same_shape(at::IntArrayRef shape, std::vector<int64_t> new_shape){
            for(int i = 0; i<shape.size(); i++){
                if(shape[i]!=new_shape[i])
                    return false;
            }
            return true;
        }
        bool reshape_top_base(at::TensorOptions options, std::vector<std::vector<int64_t>> shapes){
            if(init_top_){
                if (! is_same_shape(top_data_[0].sizes(),shapes[0])){
                    for(int i=0; i<shapes.size(); i++)
                        top_data_[i] = at::empty(shapes[i], options);
                    return true;
                }
            }else{
                for(int i=0; i<shapes.size(); i++)
                    top_data_.push_back(at::empty(shapes[i], options));
                init_top_ = true;
                return true;
            }  
            return false;
        }
		bool reshape_bottom_base(at::TensorOptions options, std::vector<std::vector<int64_t>> shapes){
            if(init_bottom_){
                if (!is_same_shape(bottom_diff_[0].sizes(),shapes[0])){
                    for(int i=0; i<shapes.size(); i++)
                        bottom_diff_[i] = at::empty(shapes[i],options);
                    return true;
                }
            }else{
                for(int i=0; i<shapes.size(); i++)
                    bottom_diff_.push_back(at::empty(shapes[i], options));
                init_bottom_ = true;
                return true;
            }  
            return false;
        }
		cudaStream_t stream_;
		std::vector<at::Tensor> top_data_;
		std::vector<at::Tensor> bottom_diff_;
		int num_, channel_, height_, width_;
		Timer* timer_;
		bool init_top_, init_bottom_;
		int  h_out_, w_out_;
		int device_=-1;
};