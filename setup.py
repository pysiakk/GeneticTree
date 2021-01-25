# copied from https://github.com/explosion/spaCy
from __future__ import print_function

import os
import subprocess
import sys
import contextlib
import numpy
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc
from distutils import ccompiler, msvccompiler
from setuptools import Extension, setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

from distutils.core import setup

PACKAGES = find_packages()

COMPILE_OPTIONS = {
    "msvc": ["/Ox", "/EHsc"],
    "mingw32": ["-O2", "-Wno-strict-prototypes", "-Wno-unused-function"],
    "other": ["-O2", "-Wno-strict-prototypes", "-Wno-unused-function"],
}


LINK_OPTIONS = {"msvc": [], "mingw32": [], "other": []}

MODULE_NAMES = [
    "genetic_tree.tree.tree",
    "genetic_tree.tree.observations",
    "genetic_tree.tree.evaluation",
    "genetic_tree.tree.thresholds",
    "genetic_tree.tree.mutator",
    "genetic_tree.tree.crosser",
    "genetic_tree.tree.builder",
    "genetic_tree.tree._utils",
]


def generate_cython(root, source):
    print("Cythonizing sources")
    p = subprocess.call(
        [sys.executable, os.path.join(root, "bin", "cythonize.py"), source],
        env=os.environ,
    )
    if p != 0:
        raise RuntimeError("Running cythonize failed")


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


class build_ext_options:
    def build_options(self):
        for e in self.extensions:
            e.extra_compile_args += COMPILE_OPTIONS.get(
                self.compiler.compiler_type, COMPILE_OPTIONS["other"]
            )
        for e in self.extensions:
            e.extra_link_args += LINK_OPTIONS.get(
                self.compiler.compiler_type, LINK_OPTIONS["other"]
            )


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


def setup_package():

    root = os.path.abspath(os.path.dirname(__file__))

    with chdir(root):
        # with io.open(os.path.join(root, "spacy", "about.py"), encoding="utf8") as f:
        #     about = {}
        #     exec(f.read(), about)

        include_dirs = [
            numpy.get_include(),
            get_python_inc(plat_specific=True),
            os.path.join(root, "include"),
        ]

        if (
            ccompiler.new_compiler().compiler_type == "msvc"
            and msvccompiler.get_build_version() == 9
        ):
            include_dirs.append(os.path.join(root, "include", "msvc9"))

        ext_modules = []
        for mod_name in MODULE_NAMES:
            mod_path = mod_name.replace(".", "/") + ".c"
            extra_link_args = []
            # ???
            # Imported from patch from @mikepb
            # See Issue #267. Running blind here...
            # if sys.platform == "darwin":
            #     dylib_path = [".." for _ in range(mod_name.count("."))]
            #     dylib_path = "/".join(dylib_path)
            #     dylib_path = "@loader_path/%s/spacy/platform/darwin/lib" % dylib_path
            #     extra_link_args.append("-Wl,-rpath,%s" % dylib_path)
            ext_modules.append(
                Extension(
                    mod_name,
                    [mod_path],
                    language="c",
                    include_dirs=include_dirs,
                    extra_link_args=extra_link_args,
                )
            )

        generate_cython(root, "genetic_tree")

        setup(
            name="genetic-tree",
            author="Tomasz Makowski & Karol Pysiak",
            author_email="tomasz.tu.tm@gmail.com",
            description="Constructing decision trees with genetic algorithm with a scikit-learn inspired API",
            long_description=long_description,
            long_description_content_type="text/markdown",
            url="https://github.com/pysiakk/GeneticTree",
            include_dirs=[numpy.get_include()],
            packages=PACKAGES,
            version="0.1.2",
            ext_modules=ext_modules,
            cmdclass={"build_ext": build_ext_subclass},
            python_requires='>=3.6',
            install_requires=[
                'aenum>=2.2.4',
                'numpy>=1.19.2',
                'scipy>=1.5.2',
            ]
        )


if __name__ == "__main__":
    setup_package()


# to build cython files run:
# python setup.py build_ext --inplace
# * run this command in root directory (GeneticTree)
# and using virtual environment
