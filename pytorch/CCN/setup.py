#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
			
cxx_args = ['-std=c++14', '-DOK']
nvcc_args = [
	'-D__CUDA_NO_HALF_OPERATORS__',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61'
]

setup(
    name='CCN',
    packages=['CCN_operator'],
    ext_modules=[
        CUDAExtension('CCN', [
            './extension/main.cpp',
            './extension/math_cuda.cu',
            './extension/dtow_cuda.cu',
            './extension/quant_cuda.cu',
			'./extension/context_reshape_cuda.cu',
			'./extension/entropy_gmm_cuda.cu',
			'./extension/mask_constrain_cuda.cu',
			'./extension/d_input_cuda.cu',
			'./extension/d_extract_cuda.cu',
			'./extension/d_output_cuda.cu',
			'./extension/dquant_cuda.cu',
			'./extension/dconv_cuda.cu',
        ],
        include_dirs=['./extension'], 
        extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}, 
        libraries=['cublas'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
