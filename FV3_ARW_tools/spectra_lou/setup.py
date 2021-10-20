#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("spectra", ["spectra.pyx"],
    extra_compile_args=['-I/Users/Louis.Wicker/miniconda3/envs/main/lib/python3.7/site-packages/numpy/core/include',
    '-I/Users/Louis.Wicker/miniconda3/envs/libs/include/c++/v1'],)]
)
