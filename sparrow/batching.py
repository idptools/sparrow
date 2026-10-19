"""Many sequences in, :class:`~sparrow.protein.Protein` objects out.

:func:`batch` is how sparrow takes a set of sequences of any size. You name the
values you want by the path you would read them back by - ``"FCR"``,
``"predictor.radius_of_gyration"``, ``"nardini.vector"`` - and get back one
``Protein`` per sequence with those values already computed, so reading them
costs nothing. Anything else is computed on demand, as for any ``Protein``.

Where sparrow has a function that computes a value for a whole batch at once,
``batch`` uses it:

* the 20 scalar properties, with :func:`sparrow.properties.compute`;
* every deep-learning prediction - the ALBATROSS polymer dimensions, DSSP,
  transmembrane regions, mitochondrial targeting, nuclear export and import
  signals, transactivation domains, PScore and phosphorylation - with the
  batch harness, :func:`~sparrow.predictors.batch_predictor.run_network`, and the
  ``polymeric.predicted_*`` values that read the ALBATROSS predictions;
* metapredict's disorder, disorder domains and pLDDT, with metapredict's own
  batch functions;
* the scrambles behind ``nardini.vector``, ``patterning``, ``gin_cluster`` and
  ``gin_report``, with :func:`sparrow.nardini.batch.nardini_proteins`, in
  several processes.

Every other value is computed one protein at a time, by the method itself, so
code that asks for it does not change if it gains a batch function later.
"""

from __future__ import annotations

import difflib
import inspect
import os
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from sparrow.predictors import BATCH_METHODS
from sparrow.properties import PROPERTY_NAMES
from sparrow.protein import Protein
from sparrow.sparrow_exceptions import SparrowException
from sparrow.tools.io import read_fasta
from sparrow.tools.utilities import cache_key

__all__ = ["batch", "ACCESSORS"]

#: The ``Protein`` accessors whose methods and properties can be named, as
#: ``"predictor.disorder"``.
ACCESSORS = ("predictor", "polymeric", "nardini")

# The device the networks run on when none is given:
# sparrow.predictors.devices.DEFAULT_DEVICE, written out so that importing sparrow
# does not import torch
_DEFAULT_DEVICE = "auto"

# Protein's public attributes that are not values to compute: the sequence
# itself, the accessors (whose own methods are named instead), the plugin
# manager and the sequence display
_NOT_VALUES = frozenset({"sequence", "show_sequence", "plugin", *ACCESSORS})

# Values that read a value a batch function computes. Asking for one of these
# computes what it reads for the whole batch first, so the value itself is cheap.
_READS = {
    "polymeric.predicted_nu": ["predictor.scaling_exponent"],
    "polymeric.predicted_rg": ["predictor.radius_of_gyration"],
    "polymeric.predicted_re": ["predictor.end_to_end_distance"],
    "polymeric.predicted_asphericity": ["predictor.asphericity"],
    "polymeric.predicted_prefactor": ["predictor.prefactor"],
    "polymeric.predicted_nu_dep_end_to_end_distribution": ["predictor.scaling_exponent", "predictor.prefactor"],
}

# The NARDINI methods that read a scrambled result, which batch computes for the
# whole batch with nardini_proteins. composition does not scramble.
_NARDINI_SCRAMBLED = ("vector", "patterning", "gin_cluster", "gin_report")

SequencesInput = str | os.PathLike[str] | Mapping[Hashable, str | Protein] | Iterable[str | Protein]
ArgumentsInput = Mapping[str, Any] | None
NamesInput = str | Iterable[str | Mapping[str, ArgumentsInput]] | Mapping[str, ArgumentsInput]


@dataclass(frozen=True)
class _Request:
    """One value asked for: where it lives on a Protein, and its arguments.

    Attributes
    ----------
    name : str
        The name as asked for, such as ``"predictor.radius_of_gyration"``.
    accessor : str or None
        One of :data:`ACCESSORS`, or None for a value of ``Protein`` itself.
    attribute : str
        The property or method, such as ``"radius_of_gyration"``.
    is_property : bool
        Whether ``attribute`` is a property, read rather than called.
    arguments : dict
        The arguments as given, which the method is called with.
    every_argument : dict
        Every argument the method is called with, defaults included. Empty for
        a property.
    """

    name: str
    accessor: str | None
    attribute: str
    is_property: bool
    arguments: dict[str, Any]
    every_argument: dict[str, Any]


def batch(
    sequences: SequencesInput,
    names: NamesInput,
    *,
    n_processes: int | None = None,
    show_progress: bool = False,
    batch_size: int | None = None,
    gpuid: int | None = None,
    device: str | Any = _DEFAULT_DEVICE,
) -> dict[Hashable, Protein] | list[Protein]:
    """Make a Protein for every sequence, with the values you name already computed.

    Each name is the path you would read the value back by, on a ``Protein``
    or one of its accessors: ``"FCR"``, ``"charge_at_pH"``,
    ``"predictor.disorder"``, ``"polymeric.mean_afrc_radius_of_gyration"``,
    ``"nardini.gin_cluster"``. Once ``batch`` returns, reading a value you
    named - with the same arguments - returns it without computing anything,
    and every other value is computed on demand, as for any ``Protein``.

    Values that sparrow can compute for a whole batch at once are computed that
    way: the 20 scalar properties, every network's and metapredict's
    predictions, and the NARDINI scrambles. The rest are computed one protein at
    a time, by the method itself.

    Parameters
    ----------
    sequences : str, os.PathLike, dict, or iterable
        The sequences: a dict of amino acid strings or ``Protein`` objects, an
        iterable of them, or the path to a FASTA file, which is read with
        ``sparrow.read_fasta`` (call that yourself to pass it options). A
        ``Protein`` you pass in is filled in and returned, not copied. A single
        sequence is not accepted, since one string is read as a path; for one
        sequence, use ``Protein(sequence)``.
    names : str, list, or dict
        The values to compute. A name on its own - ``"FCR"`` - is computed with
        its method's defaults; to give it arguments, pair it with them in a
        dict, ``{"charge_at_pH": {"pH": 6.0}}``. ``names`` can be one name, a
        list that mixes names and such dicts, or one dict, in which a name
        mapped to None uses its defaults: ``["FCR", {"charge_at_pH": {"pH": 6.0}}]``
        and ``{"FCR": None, "charge_at_pH": {"pH": 6.0}}`` ask for the same
        values. A value is then free to read with those same arguments.
    n_processes : int or None, optional
        Worker processes for the NARDINI scrambles. Default None, which uses
        every core. Nothing else runs in parallel.
    show_progress : bool, optional
        Show progress bars: for each ALBATROSS network, for the NARDINI
        scrambles, and for each value computed one protein at a time. Default
        False.
    batch_size : int or None, optional
        Sequences per forward pass of a network. Default None, which uses the
        default of :func:`~sparrow.predictors.batch_predictor.batch_predict`
        for sparrow's networks, and metapredict's own for its.
    gpuid : int or None, optional
        The NVIDIA GPU to run the networks on; see ``device``. Default None.
    device : {'auto', 'cpu', 'cuda', 'mps'} or torch.device, optional
        Where to run the networks, sparrow's and metapredict's, as PARROT's
        ``Predictor`` takes it (see
        :func:`~sparrow.predictors.devices.resolve_device`). Default
        ``'auto'``: sparrow's networks run on CUDA if this machine has it (the
        GPU ``gpuid`` names, or the current one) and on the CPU otherwise, and
        metapredict chooses for itself unless ``gpuid`` is given. A device
        named is used by both.

    Returns
    -------
    dict or list of sparrow.protein.Protein
        One ``Protein`` per sequence, in the order given: a dict with the same
        keys if ``sequences`` was a dict or a FASTA file (keyed by header), and
        a list otherwise.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If there are no sequences or no names, if an item is neither a string
        nor a ``Protein``, if a single string is not the path to a file, if a
        name is not something a ``Protein`` computes or its arguments do not
        fit, or if the device is not valid or not on this machine. Every name
        and its arguments, and the device, are checked before anything is
        computed. A value that fails to compute raises as its method does.

    Notes
    -----
    The scalar properties and NARDINI values are the ones each ``Protein``
    computes alone, to the last bit. An ALBATROSS prediction made in a batch
    can differ from one made for the sequence alone by about 1e-7 relative,
    the precision of the networks' 32-bit arithmetic, because a batch is summed
    in a different order; a prediction a ``Protein`` you pass in has already
    made is kept.

    Examples
    --------
    >>> import sparrow
    >>> proteins = sparrow.batch(["MEEEKKKKSSSTTTDDD", "GRGRGGYGGRGGYGGSRGGYGG"], ["FCR", "kappa"])
    >>> [round(protein.FCR, 3) for protein in proteins]
    [0.588, 0.182]

    >>> proteins = sparrow.batch(                                  # doctest: +SKIP
    ...     "idrs.fasta",
    ...     ["FCR", {"charge_at_pH": {"pH": 6.0}}, {"nardini.vector": {"seed": 42}}],
    ...     n_processes=8,
    ... )
    >>> proteins["IDR1"].nardini.vector(seed=42)                   # doctest: +SKIP
    """
    keys, proteins = _as_proteins(sequences)
    requests = _requests(names)
    _check_device(device, gpuid)

    # what the batch functions compute: each value asked for that one of them
    # covers, and each such value that another value asked for reads
    batched: dict[Hashable, _Request] = {}
    for request in requests:
        for read in [request] + [_request(name, {}) for name in _READS.get(request.name, [])]:
            batched.setdefault(cache_key(read.name, read.every_argument), read)

    properties = [request for request in batched.values() if _is_scalar_property(request)]
    predicted = [request for request in batched.values() if _is_predicted(request)]
    scrambled = [request for request in batched.values() if _is_scrambled(request)]

    if properties:
        _compute_properties(proteins, properties)
    if predicted:
        _predict(proteins, predicted, show_progress=show_progress, batch_size=batch_size, gpuid=gpuid, device=device)
    if scrambled:
        _score_nardini(proteins, scrambled, n_processes=n_processes, show_progress=show_progress)

    # everything else, one protein at a time; a value that reads one of those
    # above finds it already computed. A NARDINI vector is the stored result
    # itself, so it needs nothing more.
    for request in requests:
        is_vector = request.accessor == "nardini" and request.attribute == "vector"
        stored = _is_scalar_property(request) or _is_predicted(request) or is_vector
        if not stored:
            _compute_one_at_a_time(proteins, request, show_progress=show_progress)

    if keys is None:
        return proteins
    return dict(zip(keys, proteins))


# .................................................................
# what was asked for
#
def _as_proteins(sequences: SequencesInput) -> tuple[list[Hashable] | None, list[Protein]]:
    """The sequences as Protein objects, and their keys if they came in a dict.

    Parameters
    ----------
    sequences : str, os.PathLike, dict, or iterable
        As for :func:`batch`.

    Returns
    -------
    keys : list or None
        The keys, in order, or None if ``sequences`` was not a dict or a file.
    proteins : list of sparrow.protein.Protein
        One per sequence, in order; a ``Protein`` given is used as it is.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a single string is not the path to a file, there are no sequences,
        or an item is neither a string nor a ``Protein``.
    """
    if isinstance(sequences, (str, os.PathLike)):
        path = os.fspath(sequences)
        if not os.path.isfile(path):
            raise SparrowException(
                f"{path!r} is not a file. sparrow.batch reads a single string as the path to a FASTA file; "
                "for one sequence, use Protein(sequence)."
            )
        sequences = read_fasta(path)

    keys: list[Hashable] | None
    if isinstance(sequences, Mapping):
        items = list(sequences.items())
        keys = [key for key, _ in items]
        values = [value for _, value in items]
    else:
        keys = None
        values = list(sequences)

    if not values:
        raise SparrowException("No sequences were given to sparrow.batch.")

    labels: Iterable[Hashable] = keys if keys is not None else range(len(values))
    proteins = []
    for label, value in zip(labels, values):
        if isinstance(value, Protein):
            proteins.append(value)
        elif isinstance(value, str):
            proteins.append(Protein(value))
        else:
            raise SparrowException(
                f"Item {label!r} is a {type(value).__name__}; sparrow.batch takes amino acid sequences as strings, "
                "or Protein objects."
            )
    return keys, proteins


def _requests(names: NamesInput) -> list[_Request]:
    """Every value asked for, each checked, in the order asked for.

    Parameters
    ----------
    names : str, list, or dict
        As for :func:`batch`.

    Returns
    -------
    list of _Request
        One per value; a name asked for twice with the same arguments appears
        once, and with different arguments, once for each.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If there are no names, an entry of a list is neither a name nor a dict,
        or any name or its arguments is not valid.
    """
    items: list[tuple[Hashable, Mapping[str, Any]]] = []
    if isinstance(names, str):
        items.append((names, {}))
    elif isinstance(names, Mapping):
        items.extend((name, _arguments(name, arguments)) for name, arguments in names.items())
    else:
        for entry in names:
            if isinstance(entry, str):
                items.append((entry, {}))
            elif isinstance(entry, Mapping):
                items.extend((name, _arguments(name, arguments)) for name, arguments in entry.items())
            else:
                raise SparrowException(
                    "Each entry of names must be a name, such as 'FCR', or a dict of names and their arguments, "
                    f"such as {{'charge_at_pH': {{'pH': 6.0}}}}; got a {type(entry).__name__}."
                )

    if not items:
        raise SparrowException("No names were given to sparrow.batch, so there is nothing to compute.")

    # the same name with the same arguments, however they were written, is one value
    requests: dict[Hashable, _Request] = {}
    for name, arguments in items:
        request = _request(name, arguments)
        requests.setdefault(cache_key(request.name, request.every_argument), request)
    return list(requests.values())


def _arguments(name: Hashable, arguments: ArgumentsInput) -> Mapping[str, Any]:
    """The arguments given for a name, where None means its method's defaults.

    Parameters
    ----------
    name : Hashable
        The name, for the error message.
    arguments : dict or None
        Its arguments, as given.

    Returns
    -------
    dict
        The arguments; empty for None.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``arguments`` is neither a dict nor None.
    """
    if arguments is None:
        return {}
    if not isinstance(arguments, Mapping):
        raise SparrowException(
            f"The arguments for {name!r} must be a dict, such as {{'pH': 6.0}}, or None for its method's defaults; "
            f"got a {type(arguments).__name__}."
        )
    return arguments


def _request(name: Hashable, arguments: Mapping[str, Any]) -> _Request:
    """Check one name and its arguments against the property or method it names.

    Parameters
    ----------
    name : Hashable
        A name, such as ``"FCR"`` or ``"predictor.radius_of_gyration"``; anything
        but a string is refused.
    arguments : dict
        Its arguments.

    Returns
    -------
    _Request

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the name is not a string or not something a ``Protein`` computes, a
        property is given arguments, a method is given ``recompute``, or the
        arguments do not fit the method's signature.
    """
    if not isinstance(name, str):
        raise SparrowException(f"Each name must be a string, such as 'FCR'; got a {type(name).__name__}.")

    parts = name.split(".")
    if len(parts) == 1:
        accessor, attribute = None, parts[0]
    elif len(parts) == 2 and parts[0] in ACCESSORS:
        accessor, attribute = parts
    else:
        _unknown(name)

    if attribute.startswith("_") or (accessor is None and attribute in _NOT_VALUES):
        _unknown(name)

    member = inspect.getattr_static(_owner_class(accessor), attribute, None)
    if isinstance(member, property):
        if arguments:
            raise SparrowException(f"{name} is a property, so it takes no arguments; got {dict(arguments)}.")
        return _Request(name, accessor, attribute, True, {}, {})

    if not inspect.isfunction(member):
        _unknown(name)

    if "recompute" in arguments:
        raise SparrowException(f"sparrow.batch computes each value once, so {name} cannot take 'recompute'.")

    signature = inspect.signature(member)
    try:
        # None stands in for the instance the method is called on
        bound = signature.bind(None, **arguments)
    except TypeError as error:
        raise SparrowException(
            f"{name} cannot be called with {dict(arguments)}: {error}. Give a method's arguments in a dict, "
            f"such as {{{name!r}: {{...}}}}."
        ) from error
    bound.apply_defaults()

    every_argument = dict(bound.arguments)
    del every_argument[next(iter(signature.parameters))]
    return _Request(name, accessor, attribute, False, dict(arguments), every_argument)


def _owner_class(accessor: str | None) -> type:
    """The class whose property or method a name refers to.

    The accessor classes are imported only when a name refers to them, as
    ``Protein`` does, so that importing sparrow stays fast.

    Parameters
    ----------
    accessor : str or None
        One of :data:`ACCESSORS`, or None for ``Protein`` itself.

    Returns
    -------
    type
    """
    if accessor is None:
        return Protein
    if accessor == "predictor":
        from sparrow.predictors import Predictor

        return Predictor
    if accessor == "polymeric":
        from sparrow.polymer import Polymeric

        return Polymeric

    from sparrow.nardini.protein_interface import ProteinNardini

    return ProteinNardini


def _available_names() -> list[str]:
    """Every name :func:`batch` accepts, sorted.

    Returns
    -------
    list of str
        Each public property and method of ``Protein`` that is a value, and of
        each accessor, as ``"accessor.method"``.
    """
    names = []
    for accessor in (None, *ACCESSORS):
        owner = _owner_class(accessor)
        for attribute in dir(owner):
            if attribute.startswith("_") or (accessor is None and attribute in _NOT_VALUES):
                continue
            member = inspect.getattr_static(owner, attribute)
            if isinstance(member, property) or inspect.isfunction(member):
                names.append(attribute if accessor is None else f"{accessor}.{attribute}")
    return sorted(names)


def _unknown(name: str) -> NoReturn:
    """Raise for a name that is not something a Protein computes, suggesting close ones.

    Parameters
    ----------
    name : str
        The name.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        Always.
    """
    suggestions = difflib.get_close_matches(name, _available_names(), n=3)
    hint = f" Did you mean {' or '.join(repr(suggestion) for suggestion in suggestions)}?" if suggestions else ""
    raise SparrowException(
        f"{name!r} is not something a Protein computes.{hint} A name is a Protein property or method, such as "
        f"'FCR', or one of {list(ACCESSORS)} and its method, such as 'predictor.disorder'."
    )


def _is_scalar_property(request: _Request) -> bool:
    """Whether :func:`sparrow.properties.compute` computes this value.

    Parameters
    ----------
    request : _Request
        The value.

    Returns
    -------
    bool
        True if it is a scalar property.
    """
    return request.accessor is None and request.attribute in PROPERTY_NAMES


def _is_predicted(request: _Request) -> bool:
    """Whether this is a deep-learning prediction, which is made for the whole batch at once.

    Parameters
    ----------
    request : _Request
        The value.

    Returns
    -------
    bool
        True if it is a ``Predictor`` method that reads a network's prediction
        or metapredict's.
    """
    return request.accessor == "predictor" and request.attribute in BATCH_METHODS


def _is_scrambled(request: _Request) -> bool:
    """Whether this value reads a scrambled NARDINI result.

    Parameters
    ----------
    request : _Request
        The value.

    Returns
    -------
    bool
        True if it is a NARDINI method that scrambles.
    """
    return request.accessor == "nardini" and request.attribute in _NARDINI_SCRAMBLED


# .................................................................
# computing it
#
def _compute_properties(proteins: list[Protein], requests: list[_Request]) -> None:
    """Compute scalar properties for the whole batch, and store each protein's values.

    Parameters
    ----------
    proteins : list of sparrow.protein.Protein
        The batch.
    requests : list of _Request
        Scalar properties.
    """
    from sparrow.properties import compute

    table = compute([protein.sequence for protein in proteins], [request.attribute for request in requests])
    for name, values in table.items():
        for protein, value in zip(proteins, values.tolist()):
            protein._store(name, value)


def _check_device(device: str | Any, gpuid: int | None) -> None:
    """Check the device and GPU asked for, before anything is computed.

    The default, ``'auto'`` with no GPU named, is always valid, so it is not
    checked, which keeps torch from being imported for a batch with no
    predictions.

    Parameters
    ----------
    device : str or torch.device
        As :func:`batch` was given it.
    gpuid : int or None
        As :func:`batch` was given it.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the device is not valid or not on this machine.
    """
    if isinstance(device, str) and device == _DEFAULT_DEVICE and gpuid is None:
        return

    from sparrow.predictors.devices import resolve_device

    resolve_device(device, gpuid)


def _predict(
    proteins: list[Protein],
    requests: list[_Request],
    *,
    show_progress: bool,
    batch_size: int | None,
    gpuid: int | None,
    device: str | Any,
) -> None:
    """Make the predictions the requested methods read for the whole batch, and store each protein's.

    Each ``Predictor`` method says which prediction it reads for a sequence -
    a network's, or metapredict's - and how to make that for many sequences at
    once (:meth:`~sparrow.predictors.Predictor._source`). The radius of
    gyration and the end-to-end distance read a different network for a short
    sequence than for a long one, so each prediction is made once, for every
    protein that reads it, and stored where its methods look. A prediction a
    protein already has is not made again.

    Parameters
    ----------
    proteins : list of sparrow.protein.Protein
        The batch.
    requests : list of _Request
        ``Predictor`` methods that read a prediction.
    show_progress : bool
        Show a progress bar for each prediction.
    batch_size, gpuid : int or None
        Passed on unless None.
    device : str or torch.device
        Passed on, as :func:`batch` was given it.
    """
    # each prediction, with how to make it and the proteins that need it
    pending: dict[Hashable, tuple[Any, list[Protein]]] = {}
    for request in requests:
        for protein in proteins:
            source = protein.predictor._source(request.attribute, request.every_argument)
            if source is None or protein.predictor._has(source.key):
                continue
            members = pending.setdefault(source.key, (source, []))[1]
            # a protein given twice, or asked for twice, needs the prediction once
            if not any(member is protein for member in members):
                members.append(protein)

    for key, (source, members) in pending.items():
        predictions = source.predict_many(
            [protein.sequence for protein in members],
            batch_size=batch_size,
            gpuid=gpuid,
            show_progress=show_progress,
            device=device,
        )
        for protein, prediction in zip(members, predictions):
            protein.predictor._store(key, prediction)


def _score_nardini(
    proteins: list[Protein],
    requests: list[_Request],
    *,
    n_processes: int | None,
    show_progress: bool,
) -> None:
    """Compute the scrambled NARDINI results the requested methods read, for the whole batch.

    Each distinct set of options is scrambled once, by
    :func:`~sparrow.nardini.batch.nardini_proteins`, which stores each
    protein's result where its methods look.

    Parameters
    ----------
    proteins : list of sparrow.protein.Protein
        The batch.
    requests : list of _Request
        NARDINI methods that read a scrambled result.
    n_processes : int or None
        Worker processes, as for ``nardini_proteins``.
    show_progress : bool
        Show a progress bar over the chunks of each set of options.
    """
    from sparrow.nardini.batch import nardini_proteins
    from sparrow.nardini.protein_interface import ProteinNardini

    option_sets: dict[Hashable, dict[str, Any]] = {}
    for request in requests:
        options = ProteinNardini._result_options(request.attribute, request.every_argument)
        option_sets.setdefault(ProteinNardini._cache_key(**options), options)

    for options in option_sets.values():
        nardini_proteins(proteins, n_processes=n_processes, show_progress=show_progress, on_error="raise", **options)


def _compute_one_at_a_time(proteins: list[Protein], request: _Request, *, show_progress: bool) -> None:
    """Compute one value for each protein in turn, by its own property or method, which caches it.

    Parameters
    ----------
    proteins : list of sparrow.protein.Protein
        The batch.
    request : _Request
        The value.
    show_progress : bool
        Show a progress bar over the proteins.
    """
    iterator: Iterable[Protein] = proteins
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(proteins, desc=request.name)

    for protein in iterator:
        owner = protein if request.accessor is None else getattr(protein, request.accessor)
        value = getattr(owner, request.attribute)
        if not request.is_property:
            value(**request.arguments)
