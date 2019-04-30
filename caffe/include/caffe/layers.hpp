#ifndef LAYERS_HPP_
#define LAYERS_HPP_
#include"caffe/common.hpp"
#include"caffe/layers/ele_opt_layer.hpp"
namespace caffe
{
	extern INSTANTIATE_CLASS(EleOptLayer);
	REGISTER_LAYER_CLASS(EleOpt);
}
#endif