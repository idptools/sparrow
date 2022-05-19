"""
sparrow
Next generation package for sequence parameter calculation
"""

import os

# Add imports here
from .tools import io
from .protein import Protein
from .tools.io import read_fasta



# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

# code that allows access to the data directory
_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)

