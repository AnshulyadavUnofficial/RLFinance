from setuptools import setup, Extension
import pybind11

ext = Extension(
    "_cppneat",               # the name of the generated module
    ["activate.cpp"],         # your source file
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-std=c++17"],
)

setup(
    name="cppneat",
    version="0.1",
    ext_modules=[ext],
    zip_safe=False,
)

# run this line to setup setup.py
# python setup.py build_ext --inplace
