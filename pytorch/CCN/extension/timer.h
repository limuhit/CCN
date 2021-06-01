#pragma once
#include <cuda_runtime.h>
#include <string>

class Timer{
    public:
        Timer(int device = 0, bool time_it = false){
            flag_ = time_it;
            to(device);
        }
        void init(){
            if(flag_){
                cudaSetDevice(device_);
                cudaEventCreate(&start_t);
                cudaEventCreate(&stop_t);
            } 
        }
        void to(int device){
            if(device_ == device)
                return;
            device_ = device;
            init();
        }
        void set_flag(bool flag){
            if(flag_ == flag)
                return;
            flag_ = flag;
            init();
        }
        void start(){
            if(flag_){ 
                cudaEventRecord(start_t,0);
            }       
        }
        void stop(const char * head_string = ""){
            if(flag_){
                float elapsedTime;
                cudaEventRecord(stop_t,0);
	            cudaEventSynchronize(stop_t);
	            cudaEventElapsedTime(&elapsedTime, start_t,stop_t);
	            printf("%s Elapsed time : %f ms\n" ,head_string, elapsedTime);
            }    
        }
        cudaEvent_t start_t, stop_t;
        bool flag_;
        int device_=-1;
};