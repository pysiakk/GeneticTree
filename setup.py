from setuptools import Extension
from Cython.Build import cythonize
import numpy

from distutils.core import setup

defs = [('CYTHON_TRACE_NOGIL', 1)]

ext_modules = cythonize([
    Extension("tree.tree", ["tree/tree.pyx"],      # probably it has to have first name (tree)
                                                   # the same as second (tree(.pyx))
              extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension("tree.forest", ["tree/forest.pyx"],
              extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension("tree.evaluation", ["tree/evaluation.pyx"],
              extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension("tree.crosser", ["tree/crosser.pyx"],
              define_macros=defs,
              extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension("tree.builder", ["tree/builder.pyx"],
              extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension("tree._utils", ["tree/_utils.pyx"],
              define_macros=defs,
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
