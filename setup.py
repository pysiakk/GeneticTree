from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = cythonize([
    Extension("tree", ["tree.pyx"]),    # probably it has to have first name (tree)
                                        # the same as second (tree(.pyx))
    Extension("_utils", ["_utils.pyx"]),
])

setup(
    name='Genetic',
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)

# to build cython files run:
# python setup.py build_ext --inplace
# * run this command in root directory (GeneticTree)
# and using virtual environment
