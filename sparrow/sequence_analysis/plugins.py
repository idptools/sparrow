import importlib
from abc import ABC, abstractmethod
from typing import Any


class PluginManager:
    def __init__(self, protein: "sparrow.Protein"):
        self.__protein_obj = protein
        self.__precomputed = {}
        self.__plugins = {}

    def __getattr__(self, name: str):
        """
        Dynamically load and return the plugin's calculate method result.

        Parameters
        --------------
        name : str
            The name of the plugin to be accessed

        Returns
        -------------
        float
            Returns the result of the plugin's calculate method
        """
        if name not in self.__plugins:
            try:
                module = importlib.import_module(
                    f"sparrow.sequence_analysis.community_plugins.contributed"
                )
                plugin_class = getattr(module, name)
                self.__plugins[name] = plugin_class(protein=self.__protein_obj)
            except (ModuleNotFoundError, AttributeError):
                raise AttributeError(
                    f"Plugin '{name}' not found. Available plugins are: {list(self.__plugins.keys())}"
                )

        if name not in self.__precomputed:
            self.__precomputed[name] = self.__plugins[name].calculate(
                self.__protein_obj
            )

        return self.__precomputed[name]


class BasePlugin(ABC):
    """Base class for all community contributed plugins."""

    def __init__(self, protein: "sparrow.Protein"):
        """Constructor for all plugins. This must provide a protein object or sequence."""
        self.__protein_obj = protein

    @abstractmethod
    def calculate(self, protein: "sparrow.Protein") -> Any:
        """
        This method takes a sparrow.Protein object as input and must operate
        on the sequence attribute of the object. The method must return
        the result of the contributed analysis.

        Parameters
        --------------
        seq : sparrow.Protein
            A sparrow.Protein object instance

        Returns
        -------------
        float
            Returns the result of the contributed analysis

        """
        pass

    @property
    def protein(self):
        return self.__protein_obj
