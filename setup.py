"""
sparrow
Next generation package for sequence parameter calculation

sparrow is pure Python. It used to ship Cython extensions for the patterning
metrics; those were replaced by vectorized NumPy implementations in
sparrow/patterning/vectorized.py, so no compiler is needed to install.

Everything else is declared in pyproject.toml; this file only remains so that
`python setup.py ...` invocations keep working.
"""

from setuptools import setup

setup()
