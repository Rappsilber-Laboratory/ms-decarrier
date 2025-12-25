from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="ms_decarrier.decarry_batch",
        sources=["ms_decarrier/decarry_batch.pyx"],
        language="c++",                   # Required for libcpp.vector
        include_dirs=[np.get_include()],  # Required for cimport numpy
        extra_compile_args=["-O3"],       # Optimization flag
    )
]

setup(
    name="ms_utils",
    ext_modules=cythonize(extensions, language_level="3"),
)