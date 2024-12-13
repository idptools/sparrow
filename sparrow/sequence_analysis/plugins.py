import importlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any


class PluginManager:
    def __init__(self, protein: "sparrow.Protein"):
        self.__protein_obj = protein
        # Memoization for both args and no-args results
        self.__precomputed = defaultdict(dict)
        self.__plugins = {}

    def __getattr__(self, name: str):
        """
        Dynamically load and return the plugin's calculate method result
        as if it were a property when accessed without arguments.
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

        plugin_instance = self.__plugins[name]

        # Wrapper class to handle both args and no-args cases
        class PluginWrapper:
            def __init__(self, cache_dict):
                self.cache_dict = cache_dict
                self.plugin_instance = plugin_instance

            def __call__(self, *args, **kwargs):
                """
                Call calculate() with or without arguments.
                Implement caching to avoid recomputation.
                """
                # Create hashable cache key for args and kwargs
                cache_key = (args, frozenset(kwargs.items()))

                # Check if the result is cached
                if cache_key not in self.cache_dict[name]:
                    self.cache_dict[name][cache_key] = self.plugin_instance.calculate(
                        *args, **kwargs
                    )

                return self.cache_dict[name][cache_key]

        return PluginWrapper(self.__precomputed)


class BasePlugin(ABC):
    """Base class for all community contributed plugins."""

    def __init__(self, protein: "sparrow.Protein"):
        """Constructor for all plugins. This must provide a protein object or sequence."""
        self.__protein_obj = protein

    @abstractmethod
    def calculate(self) -> Any:
        """
        This method must operate on the sequence attribute of the protein object.
        The method must return the result of the contributed analysis.

        Returns
        -------------
        float
            Returns the result of the contributed analysis
        """
        pass

    @property
    def protein(self):
        return self.__protein_obj
