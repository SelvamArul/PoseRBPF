from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sdf_layer',
    ext_modules=[
        CUDAExtension(
            name='sdf_layer_cuda',
            sources=['sdf_layers.cpp',
                     'sdf_matching_loss_kernel.cu'],
            include_dirs= ['/opt/eigen_3_3_8/include/eigen3', '/usr/local/include', '/opt/sophus/include'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
