import pytest

from sparrow.protein import Protein
from sparrow.sequence_analysis.community_plugins.contributed import MultiplicativeFCR
from sparrow.sequence_analysis.plugins import BasePlugin


@pytest.fixture
def protein():
    sequence = "LLERYIPKHQKCLTSAQRSSIDPLDIEDVYQHKKPKFSSKSHIWHVYNENSNRQKLEHVKVNKGSKASLFINKEDVYEYYQKDPKNTKFGKSKHKQSTLDQIYSTGLRKGNLHNVKDPNTNVPKGIGRRKTQHKRTQVDDVDCNPRKILAVSPSRRINRLVTYQQHIPETHNDLPEELCEPSSLTLSSLRNGLDSSTEACSVSKEKHIQNLDLSDSQEVQCLELESVDQTEAVSFPGLLLHKEIKLPVVTTDKQPHTLQEQHHVLYKSHENSNLV"
    return Protein(sequence)


def test_multiplicative_fcr_plugin(protein):
    plugin_manager = protein.plugin
    double_fcr_result = plugin_manager.MultiplicativeFCR()
    expected_result = 2.0 * protein.FCR
    assert pytest.approx(double_fcr_result, 0.000001) == expected_result


def test_plugin_manager_cache(protein):
    plugin_manager = protein.plugin
    first_result = plugin_manager.MultiplicativeFCR()
    second_result = plugin_manager.MultiplicativeFCR()
    assert first_result == second_result


def test_invalid_plugin(protein):
    plugin_manager = protein.plugin
    with pytest.raises(AttributeError):
        plugin_manager.NonExistentPlugin


def test_multiple_plugins(protein):
    class TripleFCR(BasePlugin):
        def calculate(self, factor=3.0):
            return factor * self.protein.FCR

    class QuadrupleFCR(BasePlugin):
        def calculate(self, factor=4.0):
            return factor * self.protein.FCR

    plugin_manager = protein.plugin
    # plugin_manager._PluginManager__plugins is a dictionary that stores plugins.
    # we can add a new plugin to it by assigning a new key-value pair to it.
    plugin_manager._PluginManager__plugins["TripleFCR"] = TripleFCR(protein)
    plugin_manager._PluginManager__plugins["QuadrupleFCR"] = QuadrupleFCR(protein)

    # Testing TripleFCR plugin
    triple_fcr_result = plugin_manager.TripleFCR(factor=3.0)
    expected_triple_result = 3.0 * protein.FCR
    assert pytest.approx(triple_fcr_result, 0.000001) == expected_triple_result

    # Testing QuadrupleFCR plugin
    quadruple_fcr_result = plugin_manager.QuadrupleFCR(factor=4.0)
    expected_quadruple_result = 4.0 * protein.FCR
    assert pytest.approx(quadruple_fcr_result, 0.000001) == expected_quadruple_result


def test_base_plugin_initialization(protein):
    class TestPlugin(BasePlugin):
        def calculate(self):
            return protein.FCR

    plugin = TestPlugin(protein)
    assert plugin.protein == protein


def test_base_plugin_abstract_method(protein):
    with pytest.raises(TypeError):
        BasePlugin(protein)
