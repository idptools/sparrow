import pytest

from sparrow.protein import Protein
from sparrow.sequence_analysis.community_plugins.contributed import DoubleFCR
from sparrow.sequence_analysis.plugins import BasePlugin


@pytest.fixture
def protein():
    sequence = "LLERYIPKHQKCLTSAQRSSIDPLDIEDVYQHKKPKFSSKSHIWHVYNENSNRQKLEHVKVNKGSKASLFINKEDVYEYYQKDPKNTKFGKSKHKQSTLDQIYSTGLRKGNLHNVKDPNTNVPKGIGRRKTQHKRTQVDDVDCNPRKILAVSPSRRINRLVTYQQHIPETHNDLPEELCEPSSLTLSSLRNGLDSSTEACSVSKEKHIQNLDLSDSQEVQCLELESVDQTEAVSFPGLLLHKEIKLPVVTTDKQPHTLQEQHHVLYKSHENSNLV"
    return Protein(sequence)


def test_double_fcr_plugin(protein):
    plugin_manager = protein.plugin
    double_fcr_result = plugin_manager.DoubleFCR
    expected_result = 2.0 * protein.FCR
    assert pytest.approx(double_fcr_result, 0.000001) == expected_result


def test_plugin_manager_cache(protein):
    plugin_manager = protein.plugin
    first_result = plugin_manager.DoubleFCR
    second_result = plugin_manager.DoubleFCR
    assert first_result is second_result


def test_invalid_plugin(protein):
    plugin_manager = protein.plugin
    with pytest.raises(AttributeError):
        plugin_manager.NonExistentPlugin


def test_multiple_plugins(protein):
    class TripleFCR(DoubleFCR):
        def calculate(self, seq):
            return 3.0 * self.protein.FCR

    plugin_manager = protein.plugin
    plugin_manager._PluginManager__plugins["TripleFCR"] = TripleFCR(protein)
    triple_fcr_result = plugin_manager.TripleFCR
    expected_result = 3.0 * protein.FCR
    assert pytest.approx(triple_fcr_result, 0.000001) == expected_result


def test_base_plugin_initialization(protein):
    class TestPlugin(BasePlugin):
        def calculate(self, protein):
            return protein.FCR

    plugin = TestPlugin(protein)
    assert plugin.protein == protein


def test_base_plugin_abstract_method(protein):
    with pytest.raises(TypeError):
        BasePlugin(protein)
