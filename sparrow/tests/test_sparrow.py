"""
Unit and regression test for the sparrow package.
"""

# Import package, test suite, and other packages as needed
import sparrow
import pytest
import sys

def test_sparrow_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "sparrow" in sys.modules
