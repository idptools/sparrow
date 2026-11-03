"""``ProteinVisualizer``, which is ``Protein.visualize``: plots of one protein along its length."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sparrow.batching import NamesInput, _requests
from sparrow.sparrow_exceptions import SparrowException
from sparrow.visualize.protein_visualizer._analyses import _LOOKS, _analysis_tracks
from sparrow.visualize.protein_visualizer._inputs import (
    GROUPS,
    DomainsInput,
    HighlightsInput,
    SequenceInput,
    SitesInput,
    _checked_mapping,
    _checked_order,
    _domain_rows,
    _highlights,
    _sequence_track,
    _site_track,
)
from sparrow.visualize.proteinvis import (
    DEFAULT_DPI,
    DEFAULT_FIGURE_WIDTH_IN,
    DEFAULT_FONT_SIZE_PT,
    LinearTrack,
    Track,
    VisualProtein,
    plot_proteins,
    save_figure,
)

if TYPE_CHECKING:
    import os

    from sparrow.protein import Protein


class ProteinVisualizer:
    """Plots of one protein along its length, as ``Protein.visualize``.

    Each method draws the protein from the analyses you name, computed (and
    cached) by the protein, together with values, sites and domains of your
    own, as tracks stacked top to bottom on a residue axis, residue 1 on the
    left (see :mod:`sparrow.visualize.proteinvis`).

    Parameters
    ----------
    protein : sparrow.protein.Protein
        The protein to draw.
    """

    def __init__(self, protein: Protein) -> None:
        self._protein = protein

    def available_analyses(self) -> list[str]:
        """List the analyses sparrow knows how to draw, by the names they are asked for by.

        Any other analysis that gives a value for each residue can be drawn
        too, labelled with its name.

        Returns
        -------
        list of str
            The names, such as ``'predictor.disorder'``.
        """
        return list(_LOOKS)

    def visual_protein(
        self,
        analyses: NamesInput | None = None,
        *,
        signals: Mapping[str, npt.ArrayLike] | None = None,
        sites: Mapping[str, SitesInput] | None = None,
        domains: Mapping[str, DomainsInput] | None = None,
        tracks: Sequence[Track] | None = None,
        sequence: SequenceInput = None,
        highlights: HighlightsInput = None,
        name: str = "",
        order: Sequence[str] = GROUPS,
    ) -> VisualProtein:
        """Build the protein's tracks, ready to plot, without drawing them.

        :meth:`show_protein` and :meth:`save_protein` draw what this builds.
        Use it to change a track before drawing, or to draw several proteins
        in one figure with :func:`~sparrow.visualize.proteinvis.plot_proteins`.

        Parameters
        ----------
        analyses : str, list, dict or None, optional
            The protein's analyses to draw, named as :func:`sparrow.batch`
            names them: a name on its own, such as ``'predictor.disorder'``,
            is computed with its method's defaults, and a dict pairs names
            with their arguments, such as ``{'linear_sequence_profile':
            {'mode': 'NCPR', 'window_size': 9}}``. A list can mix the two, and
            hold the same name with different arguments. Each is one track,
            in order. :meth:`available_analyses` lists the analyses sparrow
            knows how to draw. Default None, none.
        signals : dict or None, optional
            Values of your own, one for each residue, by the label of the
            track that draws them as a line, such as ``{'Conservation':
            values}``. A residue with no value can be NaN. Default None.
        sites : dict or None, optional
            Sites of your own, by the label of the track that marks them. The
            sites are a list of residue positions, counting from 1, or of
            :class:`~sparrow.visualize.proteinvis.Site`, or a dict of
            positions and their labels, such as ``{'Mutations': {120:
            'R120W'}}``. Default None.
        domains : dict or None, optional
            Domains of your own, by the label of the track that draws them as
            boxes. Each domain is ``(start, end)``, ``(start, end, label)``,
            or a :class:`~sparrow.visualize.proteinvis.Domain` for its color
            and the rest, counting from 1 with both ends in the domain, such
            as ``{'Folded': [(6, 91, 'KIX')]}``. Domains that overlap are put
            on rows of their own beneath the first. Default None.
        tracks : list or None, optional
            Tracks of :mod:`sparrow.visualize.proteinvis` you have made
            yourself, such as a ``LinearTrack`` drawn as bars. Default None.
        sequence : bool, 'idrs', list or None, optional
            Write the sequence: ``True`` for the whole protein, ``'idrs'`` for
            each predicted IDR, or a list of ``(start, end)`` regions, each
            written within its region. Default None, none.
        highlights : 'idrs', list or None, optional
            Bands of color behind every track: ``'idrs'`` for each predicted
            IDR, or a list of ``(start, end)`` regions or
            :class:`~sparrow.visualize.proteinvis.Highlight`. Default None.
        name : str, optional
            The protein's name, written above its tracks. Default ``''``, none.
        order : sequence of str, optional
            The order the kinds of track are drawn in, top to bottom, from
            :data:`GROUPS`: ``'domains'``, ``'sites'``, ``'analyses'``,
            ``'signals'``, ``'tracks'`` and ``'sequence'``, which is the
            default order. Each kind given must be in it.

        Returns
        -------
        sparrow.visualize.proteinvis.VisualProtein
            The protein, with its tracks and highlights.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If there is nothing to draw, ``order`` leaves out a kind that was
            given or names one that is not, an analysis is not something the
            protein computes or does not give something for each residue, or
            any value, site, domain or region does not fit the protein. An
            analysis that fails to compute raises as its method does.
        """
        visual = VisualProtein(name, sequence=self._protein.sequence)

        # the order is checked first, before any analysis is computed
        given = {
            "domains": domains is not None,
            "sites": sites is not None,
            "analyses": analyses is not None,
            "signals": signals is not None,
            "tracks": tracks is not None,
            "sequence": sequence is not None and sequence is not False,
        }
        drawn = _checked_order(order, [group for group, is_given in given.items() if is_given])

        built: dict[str, list[Track]] = {}
        if domains is not None:
            built["domains"] = [
                row
                for label, items in _checked_mapping(domains, "domains").items()
                for row in _domain_rows(label, items)
            ]
        if sites is not None:
            built["sites"] = [_site_track(label, items) for label, items in _checked_mapping(sites, "sites").items()]
        if analyses is not None:
            built["analyses"] = [
                track for request in _requests(analyses) for track in _analysis_tracks(self._protein, request)
            ]
        if signals is not None:
            built["signals"] = [
                LinearTrack(label, values) for label, values in _checked_mapping(signals, "signals").items()
            ]
        if tracks is not None:
            built["tracks"] = list(tracks)
        if sequence is not None and sequence is not False:
            built["sequence"] = [_sequence_track(self._protein, sequence)]

        for group in drawn:
            for track in built[group]:
                visual.add_track(track)

        for highlight in _highlights(self._protein, highlights):
            visual.add_highlight(highlight.start, highlight.end, highlight.color, highlight.alpha)

        if not visual.tracks:
            raise SparrowException(
                "There is nothing to draw: name analyses, or give signals, sites, domains, tracks or sequence."
            )
        return visual

    def show_protein(
        self,
        analyses: NamesInput | None = None,
        *,
        signals: Mapping[str, npt.ArrayLike] | None = None,
        sites: Mapping[str, SitesInput] | None = None,
        domains: Mapping[str, DomainsInput] | None = None,
        tracks: Sequence[Track] | None = None,
        sequence: SequenceInput = None,
        highlights: HighlightsInput = None,
        name: str = "",
        order: Sequence[str] = GROUPS,
        figure_width_in: float = DEFAULT_FIGURE_WIDTH_IN,
        font_size_pt: float = DEFAULT_FONT_SIZE_PT,
        dpi: float = DEFAULT_DPI,
    ) -> tuple[Figure, list[Axes]]:
        """Draw the protein along its length, from its analyses and anything else you give.

        In a notebook the figure is shown below the cell; save it with
        :func:`~sparrow.visualize.proteinvis.save_figure`, or draw and save
        in one go with :meth:`save_protein`.

        Parameters
        ----------
        analyses : str, list, dict or None, optional
            The protein's analyses to draw, named as :func:`sparrow.batch`
            names them: a name on its own, such as ``'predictor.disorder'``,
            is computed with its method's defaults, and a dict pairs names
            with their arguments, such as ``{'linear_sequence_profile':
            {'mode': 'NCPR', 'window_size': 9}}``. A list can mix the two, and
            hold the same name with different arguments. Each is one track,
            in order. :meth:`available_analyses` lists the analyses sparrow
            knows how to draw. Default None, none.
        signals : dict or None, optional
            Values of your own, one for each residue, by the label of the
            track that draws them as a line, such as ``{'Conservation':
            values}``. A residue with no value can be NaN. Default None.
        sites : dict or None, optional
            Sites of your own, by the label of the track that marks them. The
            sites are a list of residue positions, counting from 1, or of
            :class:`~sparrow.visualize.proteinvis.Site`, or a dict of
            positions and their labels, such as ``{'Mutations': {120:
            'R120W'}}``. Default None.
        domains : dict or None, optional
            Domains of your own, by the label of the track that draws them as
            boxes. Each domain is ``(start, end)``, ``(start, end, label)``,
            or a :class:`~sparrow.visualize.proteinvis.Domain` for its color
            and the rest, counting from 1 with both ends in the domain, such
            as ``{'Folded': [(6, 91, 'KIX')]}``. Domains that overlap are put
            on rows of their own beneath the first. Default None.
        tracks : list or None, optional
            Tracks of :mod:`sparrow.visualize.proteinvis` you have made
            yourself, such as a ``LinearTrack`` drawn as bars. Default None.
        sequence : bool, 'idrs', list or None, optional
            Write the sequence: ``True`` for the whole protein, ``'idrs'`` for
            each predicted IDR, or a list of ``(start, end)`` regions, each
            written within its region. Default None, none.
        highlights : 'idrs', list or None, optional
            Bands of color behind every track: ``'idrs'`` for each predicted
            IDR, or a list of ``(start, end)`` regions or
            :class:`~sparrow.visualize.proteinvis.Highlight`. Default None.
        name : str, optional
            The protein's name, written above its tracks. Default ``''``, none.
        order : sequence of str, optional
            The order the kinds of track are drawn in, top to bottom, from
            :data:`GROUPS`: ``'domains'``, ``'sites'``, ``'analyses'``,
            ``'signals'``, ``'tracks'`` and ``'sequence'``, which is the
            default order. Each kind given must be in it.
        figure_width_in : float, optional
            The width of the figure, in inches. Default 6.5. The height
            follows from the tracks.
        font_size_pt : float, optional
            The size of every label, in points. Default 6.
        dpi : float, optional
            The figure's resolution on screen, and in a saved PNG, in dots per
            inch. Default 300.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The figure.
        axes : list of matplotlib.axes.Axes
            The axes of each track, top to bottom.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            As for :meth:`visual_protein`, or if the figure is too narrow for
            its labels.

        Warns
        -----
        sparrow.sparrow_exceptions.SparrowWarning
            If a region of the sequence is narrower than one letter.

        Examples
        --------
        >>> figure, axes = protein.visualize.show_protein(  # doctest: +SKIP
        ...     ["predictor.disorder", {"linear_sequence_profile": {"mode": "NCPR"}}],
        ...     domains={"Folded": [(6, 91, "KIX")]},
        ...     sequence="idrs",
        ...     highlights="idrs",
        ... )
        """
        visual = self.visual_protein(
            analyses,
            signals=signals,
            sites=sites,
            domains=domains,
            tracks=tracks,
            sequence=sequence,
            highlights=highlights,
            name=name,
            order=order,
        )
        figure, axes = plot_proteins(visual, figure_width_in=figure_width_in, font_size_pt=font_size_pt, dpi=dpi)
        return figure, axes[0]

    def save_protein(
        self,
        path: str | os.PathLike[str],
        analyses: NamesInput | None = None,
        *,
        signals: Mapping[str, npt.ArrayLike] | None = None,
        sites: Mapping[str, SitesInput] | None = None,
        domains: Mapping[str, DomainsInput] | None = None,
        tracks: Sequence[Track] | None = None,
        sequence: SequenceInput = None,
        highlights: HighlightsInput = None,
        name: str = "",
        order: Sequence[str] = GROUPS,
        figure_width_in: float = DEFAULT_FIGURE_WIDTH_IN,
        font_size_pt: float = DEFAULT_FONT_SIZE_PT,
        dpi: float = DEFAULT_DPI,
    ) -> None:
        """Draw the protein along its length and save the figure.

        The figure is drawn as :meth:`show_protein` draws it, saved with
        :func:`~sparrow.visualize.proteinvis.save_figure`, so its text stays
        text that Illustrator can edit, and then closed.

        Parameters
        ----------
        path : str or os.PathLike
            Where to save it. The extension sets the format: ``.pdf``,
            ``.svg``, ``.png`` and the rest matplotlib can write.
        analyses : str, list, dict or None, optional
            The protein's analyses to draw, named as :func:`sparrow.batch`
            names them: a name on its own, such as ``'predictor.disorder'``,
            is computed with its method's defaults, and a dict pairs names
            with their arguments, such as ``{'linear_sequence_profile':
            {'mode': 'NCPR', 'window_size': 9}}``. A list can mix the two, and
            hold the same name with different arguments. Each is one track,
            in order. :meth:`available_analyses` lists the analyses sparrow
            knows how to draw. Default None, none.
        signals : dict or None, optional
            Values of your own, one for each residue, by the label of the
            track that draws them as a line, such as ``{'Conservation':
            values}``. A residue with no value can be NaN. Default None.
        sites : dict or None, optional
            Sites of your own, by the label of the track that marks them. The
            sites are a list of residue positions, counting from 1, or of
            :class:`~sparrow.visualize.proteinvis.Site`, or a dict of
            positions and their labels, such as ``{'Mutations': {120:
            'R120W'}}``. Default None.
        domains : dict or None, optional
            Domains of your own, by the label of the track that draws them as
            boxes. Each domain is ``(start, end)``, ``(start, end, label)``,
            or a :class:`~sparrow.visualize.proteinvis.Domain` for its color
            and the rest, counting from 1 with both ends in the domain, such
            as ``{'Folded': [(6, 91, 'KIX')]}``. Domains that overlap are put
            on rows of their own beneath the first. Default None.
        tracks : list or None, optional
            Tracks of :mod:`sparrow.visualize.proteinvis` you have made
            yourself, such as a ``LinearTrack`` drawn as bars. Default None.
        sequence : bool, 'idrs', list or None, optional
            Write the sequence: ``True`` for the whole protein, ``'idrs'`` for
            each predicted IDR, or a list of ``(start, end)`` regions, each
            written within its region. Default None, none.
        highlights : 'idrs', list or None, optional
            Bands of color behind every track: ``'idrs'`` for each predicted
            IDR, or a list of ``(start, end)`` regions or
            :class:`~sparrow.visualize.proteinvis.Highlight`. Default None.
        name : str, optional
            The protein's name, written above its tracks. Default ``''``, none.
        order : sequence of str, optional
            The order the kinds of track are drawn in, top to bottom, from
            :data:`GROUPS`: ``'domains'``, ``'sites'``, ``'analyses'``,
            ``'signals'``, ``'tracks'`` and ``'sequence'``, which is the
            default order. Each kind given must be in it.
        figure_width_in : float, optional
            The width of the figure, in inches. Default 6.5. The height
            follows from the tracks.
        font_size_pt : float, optional
            The size of every label, in points. Default 6.
        dpi : float, optional
            The figure's resolution on screen, and in a saved PNG, in dots per
            inch. Default 300.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            As for :meth:`show_protein`.
        """
        figure, _ = self.show_protein(
            analyses,
            signals=signals,
            sites=sites,
            domains=domains,
            tracks=tracks,
            sequence=sequence,
            highlights=highlights,
            name=name,
            order=order,
            figure_width_in=figure_width_in,
            font_size_pt=font_size_pt,
            dpi=dpi,
        )
        try:
            save_figure(figure, path)
        finally:
            plt.close(figure)
