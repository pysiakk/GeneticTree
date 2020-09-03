from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

import numpy as np
from distutils.core import setup
from os.path import join, dirname

path = dirname(__file__)
defs = [('NPY_NO_DEPRECATED_API', 0)]

ext_modules = cythonize([
    Extension("tree.tree", ["tree/tree.pyx"],      # probably it has to have first name (tree)
                                                   # the same as second (tree(.pyx))
              extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension("tree.forest", ["tree/forest.pyx"],
              extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension("tree.crosser", ["tree/crosser.pyx"],
              extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension("tree.builder", ["tree/builder.pyx"],
              extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'],
                include_dirs = [
                       np.get_include(),
                       join(path, '..', '..')
                   ],
                   define_macros = defs),
    Extension("tree._utils", ["tree/_utils.pyx"],
              extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])
    ],
    compiler_directives={'language_level': "3"}
)
setup(
    name='Genetic',
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)

# to build cython files run:
# python setup.py build_ext --inplace
# * run this command in root directory (GeneticTree)
# and using virtual environment
