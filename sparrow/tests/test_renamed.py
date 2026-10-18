"""The functions that lost their get_ prefix still answer to their old names, for
one release, with a DeprecationWarning that names the new one and points at the
caller. The Polymeric methods are tested in test_polymeric.py."""

import importlib
import warnings

import pytest

from sparrow import Protein

# (module, old name, new name)
RENAMED_FUNCTIONS = [
    ("sparrow", "get_data", "data_path"),
    ("sparrow.data.aaindex", "get_property_values", "property_values"),
    ("sparrow.data.aaindex", "get_property_metadata", "property_metadata"),
    ("sparrow.sequence_analysis.phospho_isoforms", "get_phosphoisoforms", "phosphoisoforms"),
    ("sparrow.nardini.groups", "get_typeset", "resolve_typeset"),
    ("sparrow.nardini", "get_typeset", "resolve_typeset"),
]


@pytest.mark.parametrize(("module_name", "old_name", "new_name"), RENAMED_FUNCTIONS)
def test_an_old_name_is_the_new_function_with_a_deprecation_warning(module_name, old_name, new_name):
    module = importlib.import_module(module_name)
    with pytest.warns(
        DeprecationWarning, match=f"{module_name}.{old_name} has been renamed {module_name}.{new_name}"
    ) as caught:
        old_function = getattr(module, old_name)
    # the warning points at the line that used the old name, not inside sparrow
    assert caught[0].filename == __file__
    assert old_function is getattr(module, new_name)


@pytest.mark.parametrize(("module_name", "old_name", "new_name"), RENAMED_FUNCTIONS)
def test_an_old_name_can_still_be_imported(module_name, old_name, new_name):
    namespace = {}
    with pytest.warns(DeprecationWarning, match=new_name):
        exec(f"from {module_name} import {old_name}", namespace)
    assert namespace[old_name] is getattr(importlib.import_module(module_name), new_name)


@pytest.mark.parametrize("module_name", sorted({module_name for module_name, _, _ in RENAMED_FUNCTIONS}))
def test_an_unknown_name_still_raises(module_name):
    with pytest.raises(AttributeError, match="get_nothing"):
        getattr(importlib.import_module(module_name), "get_nothing")


def test_the_new_names_do_not_warn():
    import sparrow
    from sparrow.data import aaindex
    from sparrow.nardini.groups import resolve_typeset
    from sparrow.sequence_analysis.phospho_isoforms import phosphoisoforms

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sparrow.data_path("elm_classes.tsv")
        aaindex.property_values("hydropathy-kyte-1982")
        aaindex.property_metadata("hydropathy-kyte-1982")
        phosphoisoforms(Protein("MSAPTEEY"))
        resolve_typeset("8")
