"""Laying out the sequence of a sequence track, and writing it.

Each region's sequence is split into lines of blocks that fit within the
region, and the track is as tall as the region with the most lines.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import text_to_path
from matplotlib.transforms import blended_transform_factory, offset_copy

from sparrow.sparrow_exceptions import SparrowWarning
from sparrow.visualize.proteinvis._features import SequenceRegion
from sparrow.visualize.proteinvis._style import _POINTS_PER_INCH, INK_COLOR, SEQUENCE_FONT_FAMILY
from sparrow.visualize.proteinvis._tracks import SequenceTrack
from sparrow.visualize.proteinvis._visual_protein import VisualProtein

# The height of each line of sequence, in multiples of its font size, and the
# space above and below the sequence in its track, in points
_SEQUENCE_LINE_HEIGHT = 1.25
_SEQUENCE_PAD_PT = 2.0

# The space kept between a region's sequence and each side of the region, in
# points, so the letters stand clear of the edges of a highlight over it
_SEQUENCE_INSET_PT = 0.5


@dataclass(frozen=True)
class _RegionText:
    """One region's sequence, laid out in lines of blocks within the region.

    Attributes
    ----------
    region : SequenceRegion
        The region.
    characters : list of tuple
        ``(residue, x_pt, line)`` for each residue: its letter, the middle of
        its cell in points from the region's left edge, and its line, from 0.
    n_lines : int
        The number of lines.
    fits : bool
        Whether the sequence lies within the region; it does unless the
        region is narrower than one letter.
    """

    region: SequenceRegion
    characters: list[tuple[str, float, int]]
    n_lines: int
    fits: bool


def _character_pitch_pt(font_size_pt: float) -> float:
    """The width of each letter's cell in the sequence font, in points.

    Parameters
    ----------
    font_size_pt : float
        The size the sequence is written at, in points.

    Returns
    -------
    float
        The advance of one letter, the same for every letter of a monospaced font.
    """
    font = FontProperties(family=SEQUENCE_FONT_FAMILY, size=font_size_pt)
    width_pt, _, _ = text_to_path.get_text_width_height_descent("M", font, ismath=False)
    return float(width_pt)


def _line_of_blocks(n_residues: int, track: SequenceTrack, room_pt: float, pitch_pt: float) -> tuple[int, int]:
    """Work out how a region's sequence is split into lines that fit within the region.

    Parameters
    ----------
    n_residues : int
        The number of residues in the region.
    track : SequenceTrack
        The track the region belongs to.
    room_pt : float
        The width a line can take, in points.
    pitch_pt : float
        The width of one letter's cell, in points.

    Returns
    -------
    tuple of int
        The number of residues on each line, and in each block of a line.
        A line holds as many whole blocks as fit, up to the track's
        ``blocks_per_line``. Where not even one block fits, a line holds as
        many residues as fit, as one block, and at least one.
    """
    letters_that_fit = math.floor(room_pt / pitch_pt)
    if letters_that_fit < track.chars_per_block:
        residues_per_line = max(1, min(letters_that_fit, n_residues))
        return residues_per_line, residues_per_line

    # n blocks take n * block_pt, with one blank cell between each pair
    block_pt = track.chars_per_block * pitch_pt
    blocks_per_line = math.floor((room_pt + pitch_pt) / (block_pt + pitch_pt))
    if track.blocks_per_line is not None:
        blocks_per_line = min(blocks_per_line, track.blocks_per_line)
    blocks_per_line = min(blocks_per_line, math.ceil(n_residues / track.chars_per_block))
    return blocks_per_line * track.chars_per_block, track.chars_per_block


def _lay_out_region(
    region: SequenceRegion, sequence: str, track: SequenceTrack, residue_width_pt: float, pitch_pt: float
) -> _RegionText:
    """Lay out one region's sequence in lines of blocks, each line within the region.

    Parameters
    ----------
    region : SequenceRegion
        The region.
    sequence : str
        The protein's sequence.
    track : SequenceTrack
        The track the region belongs to.
    residue_width_pt : float
        The width of one residue on the x-axis, in points.
    pitch_pt : float
        The width of one letter's cell, in points.

    Returns
    -------
    _RegionText
        Where each letter goes.
    """
    residues = sequence[region.start - 1 : region.end]
    # the region spans from the left edge of its first residue to the right
    # edge of its last; the lines may take all of it but an inset at each side
    region_width_pt = len(residues) * residue_width_pt
    room_pt = region_width_pt - 2 * _SEQUENCE_INSET_PT
    residues_per_line, residues_per_block = _line_of_blocks(len(residues), track, room_pt, pitch_pt)

    # the lines are centered across the region as one column, each starting at
    # the same place so the blocks line up from one line to the next
    longest_line = min(len(residues), residues_per_line)
    blocks_in_longest_line = math.ceil(longest_line / residues_per_block)
    column_width_pt = (longest_line + blocks_in_longest_line - 1) * pitch_pt
    column_left_pt = (region_width_pt - column_width_pt) / 2

    characters = []
    for index, residue in enumerate(residues):
        line, column = divmod(index, residues_per_line)
        block = column // residues_per_block
        # each block is followed by one blank cell, and each letter sits in the middle of its cell
        x_pt = column_left_pt + (column + block + 0.5) * pitch_pt
        characters.append((residue, x_pt, line))

    n_lines = math.ceil(len(residues) / residues_per_line)
    return _RegionText(region, characters, n_lines, fits=pitch_pt <= room_pt)


def _lay_out_sequence(
    track: SequenceTrack, protein: VisualProtein, residue_width_pt: float
) -> tuple[list[_RegionText], float]:
    """Lay out the sequence of each region of a sequence track.

    Parameters
    ----------
    track : SequenceTrack
        The track.
    protein : VisualProtein
        The protein it is drawn for, which has a sequence.
    residue_width_pt : float
        The width of one residue on the x-axis, in points.

    Returns
    -------
    tuple
        The layout of each region, in order along the sequence, and the
        track's height in inches: enough for the region with the most lines.
    """
    assert protein.sequence is not None, "validate() checks a sequence track's protein has a sequence"
    pitch_pt = _character_pitch_pt(track.font_size_pt)
    regions = sorted(track.regions, key=lambda region: region.start)
    texts = [_lay_out_region(region, protein.sequence, track, residue_width_pt, pitch_pt) for region in regions]

    most_lines = max((text.n_lines for text in texts), default=1)
    line_height_pt = track.font_size_pt * _SEQUENCE_LINE_HEIGHT
    height_pt = 2 * _SEQUENCE_PAD_PT + most_lines * line_height_pt
    return texts, height_pt / _POINTS_PER_INCH


def _warn_if_sequence_spills(track: SequenceTrack, texts: Sequence[_RegionText]) -> None:
    """Warn if any region is too narrow to hold even one letter, so its sequence spills out of it.

    Parameters
    ----------
    track : SequenceTrack
        The track.
    texts : sequence of _RegionText
        The layout of each region, in order along the sequence.
    """
    too_narrow = [f"{text.region.start} to {text.region.end}" for text in texts if not text.fits]
    if too_narrow:
        warnings.warn(
            f"In sequence track {track.name!r}, the sequence spills out of each region narrower than one letter "
            f"(residues {'; '.join(too_narrow)}). Use a smaller font_size_pt, or a wider figure.",
            SparrowWarning,
            stacklevel=4,
        )


def _draw_sequence_text(ax: Axes, figure: Figure, track: SequenceTrack, texts: Sequence[_RegionText]) -> None:
    """Write each region's sequence within it, from the top of the track down.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The track's axes.
    figure : matplotlib.figure.Figure
        The figure.
    track : SequenceTrack
        The track.
    texts : sequence of _RegionText
        The layout of each region.
    """
    # x in residues, y as a fraction of the track's height, then offset in points
    residues_and_track_height = blended_transform_factory(ax.transData, ax.transAxes)
    line_height_pt = track.font_size_pt * _SEQUENCE_LINE_HEIGHT

    for text in texts:
        for residue, x_pt, line in text.characters:
            y_pt = -(_SEQUENCE_PAD_PT + (line + 0.5) * line_height_pt)
            ax.text(
                text.region.start - 0.5,
                1.0,
                residue,
                transform=offset_copy(residues_and_track_height, fig=figure, x=x_pt, y=y_pt, units="points"),
                ha="center",
                va="center",
                fontsize=track.font_size_pt,
                fontfamily=SEQUENCE_FONT_FAMILY,
                color=track.colors.get(residue, INK_COLOR),
            )
