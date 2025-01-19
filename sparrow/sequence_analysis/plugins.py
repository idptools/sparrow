import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any


class PluginWrapper:
    """
    A wrapper class for plugins that integrates with the plugin manager.

    This class is responsible for managing the execution of plugin instances
    and caching their results to avoid redundant computations. It uses a
    combination of the plugin name and the arguments passed to the plugin's
    `calculate` method to create a unique cache key for storing results.

    Attributes:
        name (str): The name of the plugin.
        cache_dict (dict): A dictionary used to store cached results.
        plugin_instance (object): An instance of the plugin to be wrapped.

    Methods:
        __call__(*args, **kwargs):
            Executes the plugin's `calculate` method with the provided arguments.
            Caches the result to avoid recomputation on subsequent calls with
            the same arguments.
    """

    def __init__(self, name, cache_dict, plugin_instance):
        self.name = name
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
        if cache_key not in self.cache_dict[self.name]:
            self.cache_dict[self.name][cache_key] = self.plugin_instance.calculate(
                *args, **kwargs
            )

        return self.cache_dict[self.name][cache_key]


class PluginManager:
    def __init__(self, protein: "sparrow.Protein"):
        self.__protein_obj = protein
        # Memoization for both args and no-args results
        self.__precomputed = defaultdict(dict)
        self.__plugins = {}

        self._available_plugins = self._discover_plugins()

    def _discover_plugins(self):
        """
        Discover all plugins available in the contributed plugin module.
        """
        plugin_module = "sparrow.sequence_analysis.community_plugins.contributed"
        try:
            module = importlib.import_module(plugin_module)
            return [
                name
                for name, obj in inspect.getmembers(module, inspect.isclass)
                if issubclass(obj, BasePlugin) and obj.__module__ == plugin_module
            ]
        except ModuleNotFoundError:
            return []

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
                if not issubclass(plugin_class, BasePlugin):
                    raise AttributeError(f"{name} is not a valid plugin.")
                self.__plugins[name] = plugin_class(protein=self.__protein_obj)
            except (ModuleNotFoundError, AttributeError):
                raise AttributeError(
                    f"Plugin '{name}' not found. Available plugins are: {list(self._available_plugins)}"
                )

        plugin_instance = self.__plugins[name]

        return PluginWrapper(name, self.__precomputed, plugin_instance)

    def __dir__(self):
        """
        Return the list of dynamically available plugins for autocompletion QoL.
        """
        return super().__dir__() + self._available_plugins


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
