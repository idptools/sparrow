"""Write the reference for every analysis :func:`sparrow.batch` can be asked for.

``sparrow.batch`` accepts the name of any property or method of a ``Protein``,
or of its ``predictor``, ``polymeric`` and ``nardini`` accessors, that computes
a value. This extension writes one entry for each, on every build, with its
arguments and their defaults, what it returns, and how ``batch`` computes it for
a whole set of sequences. The names come from the function ``batch`` itself uses
to suggest a name (``sparrow.batching._available_names``), and how each is
computed from the tests ``batch`` makes to choose a batch function, so the pages
list exactly the names ``batch`` accepts and cannot fall out of step with it.

The entries are written as ``.rst`` files in ``docs/batch/``, which the
hand-written pages there include. Each argument's type and meaning come from its
method's numpy-style docstring; cross-references in a docstring are written as
plain literals, since they are relative to the class they were written in.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx
from sphinx.ext.autodoc.mock import mock
from sphinx.util import logging

logger = logging.getLogger(__name__)

# The directory, under the docs source, the generated files are written to
OUTPUT_DIRECTORY = "batch"

# Each generated file, and the names it holds: Protein's own properties, its own
# methods, and each accessor's properties and methods
GROUP_FILES = {
    "protein_properties": "_analyses_protein_properties.rst",
    "protein_methods": "_analyses_protein_methods.rst",
    "predictor": "_analyses_predictor.rst",
    "polymeric": "_analyses_polymeric.rst",
    "nardini": "_analyses_nardini.rst",
}
SUMMARY_FILE = "_analyses_summary.rst"

# Where each owner class is documented, for links to the full documentation
DOCUMENTED_CLASS = {
    None: "sparrow.protein.Protein",
    "predictor": "sparrow.predictors.Predictor",
    "polymeric": "sparrow.polymer.Polymeric",
    "nardini": "sparrow.nardini.protein_interface.ProteinNardini",
}

# How a Protein reaches each accessor, for the read-back calls
READ_BACK_PREFIX = {
    None: "protein",
    "predictor": "protein.predictor",
    "polymeric": "protein.polymeric",
    "nardini": "protein.nardini",
}

# sparrow.batch refuses this argument, since it computes each value once
REFUSED_ARGUMENT = "recompute"

# A numpy-style section underline
UNDERLINE = re.compile(r"^\s*-{3,}\s*$")

# An interpreted-text role, such as :func:`~a.b` or :py:meth:`title <a.b>`. A
# role's target is relative to the class its docstring was written in, so here
# each is written as a literal. :math: is kept, since it links to nothing.
ROLE = re.compile(r":(?:[a-z]+:)?(?P<role>[a-z]+):`(?P<text>[^`]+)`")


@dataclass(frozen=True)
class Argument:
    """One argument of a method, as the reference shows it.

    Attributes
    ----------
    name : str
        The argument's name, with ``**`` for a catch-all of keywords.
    default : str or None
        The default, as Python writes it, or None for a required argument.
    kind : str
        The type, from the docstring, or an empty string.
    description : list of str
        The description's lines, from the docstring, dedented.
    """

    name: str
    default: str | None
    kind: str
    description: list[str]


@dataclass(frozen=True)
class Analysis:
    """One name sparrow.batch accepts, with everything its entry shows.

    Attributes
    ----------
    name : str
        The name, such as ``"predictor.disorder"``.
    accessor : str or None
        The accessor, or None for ``Protein`` itself.
    attribute : str
        The property or method.
    is_property : bool
        Whether it is a property, which takes no arguments.
    summary : str
        The first paragraph of its docstring.
    arguments : list of Argument
        Its arguments, in order, without ``recompute``.
    returns : list of str
        The Returns section of its docstring, dedented, or an empty list.
    computed : str
        How sparrow.batch computes it, as a sentence.
    batched : str
        Whether it is computed for the whole batch at once, for the summary
        table: ``"yes"``, ``"no"``, or ``"what it reads"``.
    group : str
        The key of :data:`GROUP_FILES` it belongs to.
    """

    name: str
    accessor: str | None
    attribute: str
    is_property: bool
    summary: str
    arguments: list[Argument]
    returns: list[str]
    computed: str
    batched: str
    group: str


# .................................................................
# reading a docstring
#
def _sections(docstring: str) -> dict[str, list[str]]:
    """Split a numpy-style docstring into its sections.

    Parameters
    ----------
    docstring : str
        The docstring, dedented, as :func:`inspect.getdoc` gives it.

    Returns
    -------
    dict
        Each section's title, mapped to its lines, with the text before the
        first section under ``""``.
    """
    lines = docstring.splitlines()
    sections: dict[str, list[str]] = {"": []}
    current = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        is_title = line.strip() and index + 1 < len(lines) and UNDERLINE.match(lines[index + 1])
        if is_title:
            current = line.strip()
            sections[current] = []
            index += 2
            continue
        sections[current].append(line)
        index += 1
    return sections


def _plain(text: str) -> str:
    """Write every role in a docstring as a literal, keeping :math:.

    Parameters
    ----------
    text : str
        Docstring text.

    Returns
    -------
    str
        The text, with ``:func:`~a.b``` written as ````b```` and
        ``:meth:`title <a.b>``` as ````title````.
    """

    def literal(match: re.Match[str]) -> str:
        if match.group("role") == "math":
            return match.group(0)
        text_of_role = match.group("text")
        title, _, target = text_of_role.partition(" <")
        if target:
            return f"``{title.strip()}``"
        if text_of_role.startswith("~"):
            return f"``{text_of_role[1:].split('.')[-1]}``"
        return f"``{text_of_role.lstrip('!')}``"

    return ROLE.sub(literal, text)


def _dedent(lines: list[str]) -> list[str]:
    """Remove the indentation the lines share, and blank lines at either end.

    Parameters
    ----------
    lines : list of str
        Lines of a docstring.

    Returns
    -------
    list of str
        The lines, dedented.
    """
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    shared = min(indents, default=0)
    return [line[shared:] if line.strip() else "" for line in lines]


def _summary(sections: dict[str, list[str]]) -> str:
    """The first paragraph of a docstring, as one line.

    Parameters
    ----------
    sections : dict
        The docstring's sections, from :func:`_sections`.

    Returns
    -------
    str
        The paragraph, with its roles written as literals.
    """
    paragraph: list[str] = []
    for line in sections[""]:
        if not line.strip():
            if paragraph:
                break
            continue
        paragraph.append(line.strip())
    return _plain(" ".join(paragraph))


def _described_parameters(sections: dict[str, list[str]]) -> dict[str, tuple[str, list[str]]]:
    """Each parameter a docstring describes, with its type and description.

    An entry naming several parameters, ``num_scrambles, seed`` - as one that
    says they are "as for" another method does - describes each of them.

    Parameters
    ----------
    sections : dict
        The docstring's sections, from :func:`_sections`.

    Returns
    -------
    dict
        Each parameter's name, without any ``*``, mapped to its type and its
        description's lines.
    """
    described: dict[str, tuple[str, list[str]]] = {}
    entries: list[tuple[list[str], str, list[str]]] = []
    for line in sections.get("Parameters", []):
        if line.strip() and not line.startswith((" ", "\t")):
            names, _, kind = line.partition(" :")
            if not kind:
                names, _, kind = line.partition(":")
            entries.append(([name.strip().lstrip("*") for name in names.split(",")], kind.strip(), []))
        elif entries:
            entries[-1][2].append(line)

    for names, kind, description in entries:
        for name in names:
            described[name] = (kind, _dedent(description))
    return described


# .................................................................
# what each name is
#
def _arguments(member: Callable[..., Any], described: dict[str, tuple[str, list[str]]], name: str) -> list[Argument]:
    """A method's arguments, as its signature and docstring give them.

    Parameters
    ----------
    member : callable
        The method.
    described : dict
        Its docstring's parameters, from :func:`_described_parameters`.
    name : str
        The name sparrow.batch knows it by, for warnings.

    Returns
    -------
    list of Argument
        Every argument but the instance and ``recompute``.
    """
    parameters = list(inspect.signature(member).parameters.values())[1:]
    arguments = []
    for parameter in parameters:
        if parameter.name == REFUSED_ARGUMENT:
            continue
        kind, description = described.get(parameter.name, ("", []))
        if not description:
            logger.warning("sparrow.batch reference: %s does not describe its argument %r", name, parameter.name)

        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            shown_name, default = f"**{parameter.name}", None
        elif parameter.default is inspect.Parameter.empty:
            shown_name, default = parameter.name, None
        else:
            shown_name, default = parameter.name, repr(parameter.default)
        arguments.append(Argument(shown_name, default, _plain(kind), description))
    return arguments


def _computed(name: str, accessor: str | None, attribute: str, is_property: bool) -> tuple[str, str]:
    """How sparrow.batch computes a value, as a sentence, and whether it is for the whole batch at once.

    Parameters
    ----------
    name : str
        The name, such as ``"predictor.disorder"``.
    accessor : str or None
        Its accessor, or None for ``Protein`` itself.
    attribute : str
        The property or method.
    is_property : bool
        Whether it is a property.

    Returns
    -------
    tuple of str
        The sentence, in reStructuredText, and ``"yes"``, ``"no"`` or
        ``"what it reads"``, as :attr:`Analysis.batched`.
    """
    from sparrow.batching import _NARDINI_SCRAMBLED, _READS, _Request, _is_predicted, _is_scalar_property
    from sparrow.data.configs import MIN_LENGTH_ALBATROSS_RE_RG
    from sparrow.predictors import METAPREDICT_METHODS, METHOD_NETWORKS, SCALED_NETWORK_METHODS, SCALED_NETWORK_PREFIX

    request = _Request(name, accessor, attribute, is_property, {}, {})
    if _is_scalar_property(request):
        return "For the whole batch at once, with :func:`sparrow.properties.compute`.", "yes"

    if _is_predicted(request) and attribute in METHOD_NETWORKS:
        network = METHOD_NETWORKS[attribute]
        if attribute in SCALED_NETWORK_METHODS:
            scaled = SCALED_NETWORK_PREFIX + network
            network_text = (
                f"the ``{scaled}`` network (``{network}`` with ``use_scaled=False``, except for sequences shorter "
                f"than {MIN_LENGTH_ALBATROSS_RE_RG} residues, which use ``{scaled}`` unless ``safe=False``)"
            )
        else:
            network_text = f"the ``{network}`` network"
        sentence = (
            f"For the whole batch at once: {network_text} runs over every sequence in batches, once however many "
            "of its methods are asked for, and each protein's prediction is made from its output."
        )
        return sentence, "yes"

    if _is_predicted(request) and attribute in METAPREDICT_METHODS:
        if attribute == "pLDDT":
            return "For the whole batch at once, with metapredict's ``predict_pLDDT``.", "yes"
        if attribute == "binary_disorder":
            sentence = (
                "For the whole batch at once, with metapredict's ``predict_disorder``, once for each set of options."
            )
            return sentence, "yes"
        sentence = (
            "For the whole batch at once, with metapredict's ``predict_disorder``, once for "
            "``predictor.disorder`` and ``predictor.disorder_domains`` together."
        )
        return sentence, "yes"

    if name in _READS:
        read = " and ".join(f"``{read_name}``" for read_name in _READS[name])
        return f"One protein at a time, from {read}, which is predicted for the whole batch first.", "what it reads"

    if accessor == "nardini" and attribute in _NARDINI_SCRAMBLED:
        sentence = (
            "For the whole batch at once: the scrambles run in ``n_processes`` processes, with "
            ":func:`~sparrow.nardini.batch.nardini_proteins`, once for each set of options."
        )
        return sentence, "yes"

    return "One protein at a time, by the method itself.", "no"


def _group(accessor: str | None, is_property: bool) -> str:
    """The key of :data:`GROUP_FILES` a name belongs to.

    Parameters
    ----------
    accessor : str or None
        Its accessor, or None for ``Protein`` itself.
    is_property : bool
        Whether it is a property.

    Returns
    -------
    str
    """
    if accessor is None:
        return "protein_properties" if is_property else "protein_methods"
    return accessor


def collect_analyses() -> list[Analysis]:
    """Every name sparrow.batch accepts, with everything its entry shows.

    Returns
    -------
    list of Analysis
        In the order of :data:`GROUP_FILES`, and by name within each group.
    """
    from sparrow.batching import _available_names, _owner_class

    analyses = []
    for name in _available_names():
        accessor_text, _, attribute = name.rpartition(".")
        accessor = accessor_text or None
        member = inspect.getattr_static(_owner_class(accessor), attribute)
        is_property = isinstance(member, property)
        function = member.fget if is_property else member

        sections = _sections(inspect.getdoc(function) or "")
        if not sections[""] or not _summary(sections):
            logger.warning("sparrow.batch reference: %s has no docstring", name)

        arguments = [] if is_property else _arguments(function, _described_parameters(sections), name)
        computed, batched = _computed(name, accessor, attribute, is_property)
        analyses.append(
            Analysis(
                name=name,
                accessor=accessor,
                attribute=attribute,
                is_property=is_property,
                summary=_summary(sections),
                arguments=arguments,
                returns=[_plain(line) for line in _dedent(sections.get("Returns", []))],
                computed=computed,
                batched=batched,
                group=_group(accessor, is_property),
            )
        )

    order = list(GROUP_FILES)
    return sorted(analyses, key=lambda analysis: (order.index(analysis.group), analysis.name.lower()))


# .................................................................
# writing the entries
#
def label(name: str) -> str:
    """The reference label of a name's entry.

    Parameters
    ----------
    name : str
        The name, such as ``"predictor.disorder"``.

    Returns
    -------
    str
        Such as ``"batch-analysis-predictor-disorder"``.
    """
    return "batch-analysis-" + name.replace(".", "-").replace("_", "-").lower()


def _full_documentation(analysis: Analysis) -> str:
    """A link to where the property or method is documented in full.

    Parameters
    ----------
    analysis : Analysis

    Returns
    -------
    str
        A ``:attr:`` or ``:meth:`` role.
    """
    role = "attr" if analysis.is_property else "meth"
    return f":{role}:`{DOCUMENTED_CLASS[analysis.accessor]}.{analysis.attribute}`"


def _read_back(analysis: Analysis) -> str:
    """How to read the value back from a Protein, with every default.

    Parameters
    ----------
    analysis : Analysis

    Returns
    -------
    str
        Such as ``protein.charge_at_pH(pH=7.4, normalize=True)``.
    """
    prefix = f"{READ_BACK_PREFIX[analysis.accessor]}.{analysis.attribute}"
    if analysis.is_property:
        return prefix

    shown = []
    for argument in analysis.arguments:
        if argument.name.startswith("**"):
            shown.append(argument.name)
        elif argument.default is None:
            shown.append(f"{argument.name}=...")
        else:
            shown.append(f"{argument.name}={argument.default}")
    return f"{prefix}({', '.join(shown)})"


def _indented(lines: list[str], indent: str) -> list[str]:
    """Indent each non-blank line.

    Parameters
    ----------
    lines : list of str
    indent : str

    Returns
    -------
    list of str
    """
    return [indent + line if line.strip() else "" for line in lines]


def _entry(analysis: Analysis) -> list[str]:
    """One name's entry.

    Parameters
    ----------
    analysis : Analysis

    Returns
    -------
    list of str
        The entry's reStructuredText lines.
    """
    title = f'``"{analysis.name}"``'
    lines = [f".. _{label(analysis.name)}:", "", title, "^" * len(title), ""]
    if analysis.summary:
        lines += [analysis.summary, ""]

    with_arguments = "" if analysis.is_property or not analysis.arguments else ", with the arguments you named it with"
    lines += [
        f"* **Read it back as** ``{_read_back(analysis)}``{with_arguments}. Documented in full at "
        f"{_full_documentation(analysis)}.",
        f"* **Computed** {analysis.computed[0].lower()}{analysis.computed[1:]}",
    ]
    if analysis.is_property:
        lines.append("* **Arguments:** none, since it is a property.")
    elif not analysis.arguments:
        lines.append("* **Arguments:** none.")
    lines.append("")

    if analysis.arguments:
        lines += ["**Arguments**", ""]
        for argument in analysis.arguments:
            lines.append(_argument_heading(argument))
            lines += _indented([_plain(line) for line in argument.description] or ["Not described."], "   ")
            lines.append("")

    if analysis.returns:
        lines += ["**Returns**", ""]
        lines += _returns_block(analysis.returns)
        lines.append("")
    return lines


def _argument_heading(argument: Argument) -> str:
    """An argument's name, with its type and its default, or that it is required.

    Parameters
    ----------
    argument : Argument

    Returns
    -------
    str
        Such as ````pH```` ``(float, optional; default ``7.4``)``.
    """
    details = [argument.kind] if argument.kind else []
    if argument.default is not None:
        details.append(f"default ``{argument.default}``")
    elif not argument.name.startswith("**"):
        details.append("required")

    heading = f"``{argument.name}``"
    if details:
        heading += f" ({'; '.join(details)})"
    return heading


def _argument_summary(argument: Argument) -> str:
    """An argument as the summary table shows it: with its default, or marked required.

    Parameters
    ----------
    argument : Argument

    Returns
    -------
    str
        Such as ````pH=7.4````.
    """
    if argument.name.startswith("**"):
        return f"``{argument.name}``"
    if argument.default is None:
        return f"``{argument.name}`` (required)"
    return f"``{argument.name}={argument.default}``"


def _returns_block(returns: list[str]) -> list[str]:
    """A Returns section as a definition list: each type, with its description.

    Parameters
    ----------
    returns : list of str
        The section's lines, dedented, roles already written as literals.

    Returns
    -------
    list of str
    """
    # each unindented line is a type; the indented lines after it describe it
    entries: list[tuple[str, list[str]]] = []
    for line in returns:
        if line.strip() and not line.startswith((" ", "\t")):
            entries.append((line, []))
        elif entries:
            entries[-1][1].append(line)

    lines = []
    for kind, description in entries:
        lines.append(kind)
        lines += _indented(_dedent(description), "   ")
        lines.append("")
    return lines


def _summary_table(analyses: list[Analysis]) -> list[str]:
    """One table of every name: its arguments, and whether it is computed for the whole batch.

    Parameters
    ----------
    analyses : list of Analysis

    Returns
    -------
    list of str
    """
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 38 42 20",
        "",
        "   * - Name",
        "     - Arguments, with their defaults",
        "     - For the whole batch at once?",
    ]
    for analysis in analyses:
        if analysis.is_property:
            arguments = "(a property)"
        elif not analysis.arguments:
            arguments = "none"
        else:
            arguments = ", ".join(_argument_summary(argument) for argument in analysis.arguments)
        lines += [
            f"   * - :ref:`{analysis.name} <{label(analysis.name)}>`",
            f"     - {arguments}",
            f"     - {analysis.batched}",
        ]
    return lines


def write_analysis_pages(app: Sphinx) -> None:
    """Write each group's entries, and the summary table, into ``docs/batch/``.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The application, whose source directory the files are written under.
    """
    try:
        with mock(app.config.autodoc_mock_imports):
            analyses = collect_analyses()
    except Exception as error:  # pragma: no cover - warn, but keep the rest of the docs building
        logger.warning("Could not build the sparrow.batch reference: %s", error)
        return

    output = Path(app.srcdir) / OUTPUT_DIRECTORY
    for group, filename in GROUP_FILES.items():
        lines: list[str] = [".. This file is written by docs/_ext/batch_analyses.py on every build. Do not edit.", ""]
        for analysis in analyses:
            if analysis.group == group:
                lines += _entry(analysis)
        _write_if_changed(output / filename, lines)

    summary = [".. This file is written by docs/_ext/batch_analyses.py on every build. Do not edit.", ""]
    _write_if_changed(output / SUMMARY_FILE, summary + _summary_table(analyses))


def _write_if_changed(path: Path, lines: list[str]) -> None:
    """Write a file only if its contents change, so Sphinx does not reread every page that includes it.

    Parameters
    ----------
    path : pathlib.Path
    lines : list of str
    """
    text = "\n".join(lines) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the extension.

    Parameters
    ----------
    app : sphinx.application.Sphinx

    Returns
    -------
    dict
        The extension's metadata.
    """
    app.connect("builder-inited", write_analysis_pages)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
