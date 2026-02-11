"""
sparrow
Next generation package for sequence parameter calculation
"""

import os

# Add imports here
from sparrow.tools import io
from sparrow.protein import Protein
from sparrow.tools.io import build_grammar_background_from_fasta, read_fasta


# Generate _version.py if missing and in the Read the Docs environment
if os.getenv("READTHEDOCS") == "True" and not os.path.isfile('../sparrow/_version.py'):   
    import versioningit            
    __version__ = versioningit.get_version('../')
else:
    from ._version import __version__

# code that allows access to the data directory
_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)
