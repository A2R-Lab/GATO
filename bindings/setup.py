from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import os
import torch
from pathlib import Path

cuda_include = Path(CUDA_HOME) / 'include'
cuda_lib = Path(CUDA_HOME) / 'lib64'
torch_lib_path = Path(torch.__file__).parent

# Common arguments for all extensions
common_include_dirs = [
    '.',
    '..',
    '../gato',
    '../config',
    '../dependencies',
    '/usr/include/eigen3',
    str(cuda_include),
]

common_library_dirs = [
    str(torch_lib_path),
    str(cuda_lib),
]

common_runtime_library_dirs = [
    str(torch_lib_path),
    str(cuda_lib),
]

common_extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-use_fast_math',
        '-O3',
        '-std=c++17',
    ]
}

# extensions for each knot point value
knot_point_values = [16]
extensions = []

for knot_points in knot_point_values:
    # Create a copy of the nvcc args and add the define for KNOT_POINTS
    nvcc_args = common_extra_compile_args['nvcc'].copy()
    nvcc_args.append(f'-DKNOT_POINTS={knot_points}')
    
    # Create the extension
    extensions.append(
        CUDAExtension(
            f'bsqp_N{knot_points}', 
            ['python/batch_sqp_solver.cu'],
            include_dirs=common_include_dirs,
            library_dirs=common_library_dirs,
            runtime_library_dirs=common_runtime_library_dirs,
            extra_compile_args={
                'cxx': common_extra_compile_args['cxx'],
                'nvcc': nvcc_args
            }
        )
    )

setup(
    name='bsqp',
    ext_modules=extensions,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True, build_dir='build')
    },
    py_modules=['python.bsqp_wrapper'],
    packages=['python'],
)