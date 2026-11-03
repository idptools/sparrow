"""A protein to plot: its length or sequence, the tracks drawn for it, and the residues highlighted."""

from __future__ import annotations

from sparrow.sparrow_exceptions import SparrowException
from sparrow.visualize.proteinvis._checks import _check_count
from sparrow.visualize.proteinvis._features import Highlight
from sparrow.visualize.proteinvis._style import DEFAULT_HIGHLIGHT_ALPHA, DEFAULT_HIGHLIGHT_COLOR
from sparrow.visualize.proteinvis._tracks import _TRACK_TYPES, Track


class VisualProtein:
    """A protein to plot: its length or sequence, the tracks drawn for it, and the residues highlighted.

    Parameters
    ----------
    name : str
        The protein's name, written above its tracks. ``''`` writes none.
    length : int or None, optional
        The number of residues. Default None, the length of ``sequence``.
    sequence : str or None, optional
        The sequence, which a :class:`SequenceTrack` needs. Default None.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If neither ``length`` nor ``sequence`` is given, they disagree, or
        either is not as described.
    """

    def __init__(self, name: str, length: int | None = None, sequence: str | None = None) -> None:
        if not isinstance(name, str):
            raise SparrowException(f"A protein's name must be a str, not {name!r}.")
        self.name = name

        if sequence is not None:
            if not isinstance(sequence, str) or len(sequence) == 0:
                raise SparrowException(
                    f"A protein's sequence must be a str of at least one residue, not {sequence!r}."
                )
            if length is not None and length != len(sequence):
                raise SparrowException(f"length is {length!r}, but the sequence has {len(sequence)} residues.")
            self.sequence: str | None = sequence
            self.length = len(sequence)
        elif length is not None:
            self.sequence = None
            self.length = _check_count(length, "A protein's length")
        else:
            raise SparrowException("A VisualProtein needs its length or its sequence.")

        self.tracks: list[Track] = []
        self.highlights: list[Highlight] = []

    def add_track(self, track: Track) -> None:
        """Add a track, drawn beneath those already added.

        Parameters
        ----------
        track : DomainTrack, SiteTrack, LinearTrack, GridTrack or SequenceTrack
            The track.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If ``track`` is not a track, or does not fit the protein.
        """
        if not isinstance(track, _TRACK_TYPES):
            raise SparrowException(
                "A track must be a DomainTrack, SiteTrack, LinearTrack, GridTrack or SequenceTrack, not "
                f"{type(track).__name__}."
            )
        track.validate(self)
        self.tracks.append(track)

    def add_highlight(
        self, start: int, end: int, color: str = DEFAULT_HIGHLIGHT_COLOR, alpha: float = DEFAULT_HIGHLIGHT_ALPHA
    ) -> Highlight:
        """Add a band of color behind every track of the protein, over residues ``start`` to ``end``.

        Parameters
        ----------
        start, end : int
            The first and last residue the band covers, counting from 1.
        color : str, optional
            The band's color, any color matplotlib can draw. Default red.
        alpha : float, optional
            The band's opacity, from 0 (invisible) to 1 (solid). Default 0.15.

        Returns
        -------
        Highlight
            The highlight added.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If any of these is not as described, or the band ends after the
            protein's last residue.
        """
        highlight = Highlight(start, end, color, alpha)
        if highlight.end > self.length:
            raise SparrowException(
                f"The highlight from {highlight.start} to {highlight.end} ends after the last residue of "
                f"{self.name or 'the protein'} ({self.length})."
            )
        self.highlights.append(highlight)
        return highlight

    def validate(self) -> None:
        """Check that the protein has a track, and that every track fits it.

        Tracks are checked as they are added, and again here, since a domain or
        region can be added to a track after the track is added to the protein.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the protein has no tracks, or a track does not fit it.
        """
        if not self.tracks:
            raise SparrowException(f"{self.name or 'The protein'} has no tracks to draw: add one with add_track.")
        for track in self.tracks:
            track.validate(self)
