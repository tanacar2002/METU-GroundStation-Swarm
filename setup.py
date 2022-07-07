from setuptools import setup
from Cython.Build import cythonize

setup(
    name='METU Ground Station Control',
    ext_modules=cythonize("groundstation_v2.pyx"),
    zip_safe=False,
)