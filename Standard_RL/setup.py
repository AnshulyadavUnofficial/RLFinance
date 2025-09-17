from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

cpp_source = "per_bindings.cpp"  # relative to setup.py inside StandardRL

ext_modules = [
    Pybind11Extension(
        name="per_buffer",    # Python module name
        sources=[cpp_source],
        cxx_std=17
    )
]

setup(
    name="per_buffer",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
