from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import os
import torch
from pathlib import Path

cuda_include = Path(CUDA_HOME) / 'include'
cuda_lib = Path(CUDA_HOME) / 'lib64'
torch_lib_path = Path(torch.__file__).parent

setup(
    name='batch_sqp',
    ext_modules=[
        CUDAExtension('batch_sqp', [
            'python/batch_sqp_solver.cu', 
        ],
        include_dirs=[
            '.',
            '../gato',
            '../config',
            '../dependencies',
            str(cuda_include),
        ],
        library_dirs=[
            str(torch_lib_path),
            str(cuda_lib),
        ],
        runtime_library_dirs=[
            str(torch_lib_path),
            str(cuda_lib),
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-use_fast_math',
                '-O3',
                '-std=c++17',
            ]
        })
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True, build_dir='build')
    },
)