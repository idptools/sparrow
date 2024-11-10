"""
sparrow
Next generation package for sequence parameter calculation
"""
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os
import numpy

# defines the absolute path of where your cython files are
cython_dir = os.path.join("sparrow", "patterning")

# build a list of the files 
cython_files = [os.path.join(cython_dir, f) for f in os.listdir(cython_dir) if f.endswith('.pyx')]


extensions = [
    Extension(
        name=f"sparrow.patterning.{os.path.splitext(os.path.basename(file))[0]}",
        sources=[file],
        include_dirs=[numpy.get_include()],
    ) for file in cython_files
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    packages=find_packages(),
    include_package_data=True,
)
