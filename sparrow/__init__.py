"""
sparrow
Next generation package for sequence parameter calculation
"""

# Add imports here
from .tools import io
from .protein import Protein



# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
