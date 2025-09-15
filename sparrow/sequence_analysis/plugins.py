"""Plugin infrastructure for Sparrow sequence analysis.

This module provides a lightweight dynamic plugin discovery and execution
mechanism used by Sparrow to run community contributed sequence analysis
routines. Plugins are simple subclasses of :class:`BasePlugin` that
implement a single :meth:`BasePlugin.calculate` method. They are discovered
at runtime from the ``sparrow.sequence_analysis.community_plugins.contributed``
namespace and exposed as attributes on :class:`PluginManager` instances.

Typical usage
-------------
>>> mgr = PluginManager(protein_obj)
>>> result = mgr.HydrophobicityIndex()          # call plugin with no args
>>> result2 = mgr.SomeOtherMetric(window=5)     # call plugin with args (cached)

Caching
-------
Results of plugin executions are memoized per plugin + arguments so repeated
calls with identical arguments are O(1) after the first computation.

Notes
-----
* Accessing an unknown attribute raises ``AttributeError`` listing available plugins.
* Autocompletion is improved via ``__dir__`` which returns discovered plugin names.

"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from sparrow import Protein

__all__ = ["PluginWrapper", "PluginManager", "BasePlugin"]


class PluginWrapper:
    """Callable wrapper adding argument-aware result caching for a plugin.

    The wrapper caches results keyed by the positional argument tuple and a
    frozenset of keyword argument items to avoid repeated calculations for
    identical invocations of the underlying plugin's ``calculate`` method.

    Parameters
    ----------
    name : str
        Canonical plugin name (class name).
    cache_dict : dict[str, dict[tuple, Any]]
        Shared memoization dictionary managed by :class:`PluginManager`.
    plugin_instance : BasePlugin
        Instantiated plugin object providing ``calculate``.

    Notes
    -----
    The cache key is constructed as ``(args, frozenset(kwargs.items()))`` which
    requires that all argument values be hashable.
    """

    def __init__(self, name, cache_dict, plugin_instance):
        self.name = name
        self.cache_dict = cache_dict
        self.plugin_instance = plugin_instance

    def __call__(self, *args, **kwargs):
        """Execute the wrapped plugin with memoization.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``calculate``.
        **kwargs
            Keyword arguments forwarded to ``calculate``.

        Returns
        -------
        Any
            Result returned by the plugin's ``calculate`` method (cached).
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
    """Discover, load, cache, and expose community contributed plugins.

    A :class:`PluginManager` instance behaves as a dynamic attribute container
    where each attribute access corresponding to a discovered plugin name
    returns a callable :class:`PluginWrapper`. Invoking that callable executes
    the plugin's :meth:`BasePlugin.calculate` method with transparent caching
    keyed by arguments.

    Parameters
    ----------
    protein_obj : Protein
        Protein object whose sequence (and related metadata) plugin analyses
        will operate on.

    Attributes
    ----------
    _available_plugins : list[str]
        Names of all plugin classes discovered under the contributed namespace.
    _PluginManager__protein_obj : Protein
        Stored protein instance (private attribute).
    _PluginManager__precomputed : dict[str, dict[tuple, Any]]
        Nested dictionary mapping plugin name to cached results keyed by the
        argument signature tuple used in :class:`PluginWrapper`.
    _PluginManager__plugins : dict[str, BasePlugin]
        Loaded plugin instances keyed by name.

    Notes
    -----
    * Discovery happens once at initialization.
    * Attribute access for an undiscovered plugin raises ``AttributeError`` with
      a helpful list of available plugins.
    * Autocompletion in interactive environments is aided by overriding
      :meth:`__dir__` to include discovered plugin names.
    """

    def __init__(self, protein_obj: "Protein"):
        """Initialize the manager and eagerly discover available plugins.

        Parameters
        ----------
        protein_obj : Protein
            Protein instance passed to each plugin upon first access.
        """
        self.__protein_obj = protein_obj
        # Memoization for both args and no-args results
        self.__precomputed = defaultdict(dict)
        self.__plugins = {}

        self._available_plugins = self._discover_plugins()

    def _discover_plugins(self):
        """Return a list of contributed plugin class names.

        Discovery is limited to classes that:
        * Reside directly in the contributed plugins module; and
        * Subclass :class:`BasePlugin`.

        Returns
        -------
        list[str]
            Sorted list of discovered plugin class names (may be empty).
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
        """Lazily load a plugin and return its callable wrapper.

        Parameters
        ----------
        name : str
            Plugin class name to access.

        Returns
        -------
        PluginWrapper
            Wrapper that dispatches to the plugin's ``calculate`` and caches results.

        Raises
        ------
        AttributeError
            If the named plugin cannot be found or is not a valid subclass.
        """
        if name not in self.__plugins:
            try:
                module = importlib.import_module(
                    "sparrow.sequence_analysis.community_plugins.contributed"
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
        """Return default attributes plus dynamically discovered plugin names."""
        return super().__dir__() + self._available_plugins


class BasePlugin(ABC):
    """Abstract base class for all contributed plugins.

    Subclasses must implement :meth:`calculate`, operating on the provided
    protein object's sequence to return an analysis result.

    Parameters
    ----------
    protein : Protein
        Protein instance supplied by :class:`PluginManager`.
    """

    def __init__(self, protein: "Protein"):
        self.__protein_obj = protein

    @abstractmethod
    def calculate(self) -> Any:
        """Run the plugin's analysis logic.

        Returns
        -------
        Any
            Result of the contributed analysis (type is plugin-specific).
        """
        pass

    @property
    def protein(self):
        """Return the protein instance associated with this plugin."""
        return self.__protein_obj
