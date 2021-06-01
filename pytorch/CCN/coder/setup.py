#!/usr/bin/env python3
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

cxx_args = ['-std=c++11','/wd4251', '/W0', '/O1']

setup(
    name='coder',
    ext_modules=[
        CppExtension('coder', [
            'python.cpp',
            'BitIoStream.cpp',
            'ArithmeticCoder.cpp'
        ],
        extra_compile_args={'cxx': cxx_args})],
    cmdclass={
        'build_ext': BuildExtension
    })
