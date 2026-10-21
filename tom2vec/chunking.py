from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

DEFAULT_SENTENCE_DELIMITERS: Tuple[str, ...] = (".", "!", "?", "\n")

DEFAULT_K_LATENTS = 10
DEFAULT_N_CHUNK_LATENTS = DEFAULT_K_LATENTS - 2


@dataclass
class ChunkReadout:
    """Where to read the chunk-aligned latents out of a generated trace."""

    positions: List[int]
    boundaries: List[int] = field(default_factory=list)
    n_chunks: int = 0
    n_filled: int = 0
    degenerate: bool = False


def find_chunk_boundaries(
    token_pieces: Sequence[str],
    delimiters: Sequence[str] = DEFAULT_SENTENCE_DELIMITERS,
) -> List[int]:
    delims = tuple(delimiters)
    collapsed: List[int] = []
    for i, piece in enumerate(token_pieces):
        if not (piece and any(d in piece for d in delims)):
            continue
        if collapsed and i == collapsed[-1] + 1:
            collapsed[-1] = i
        else:
            collapsed.append(i)
    return collapsed


def _even_grid(n_tokens: int, n_points: int) -> List[int]:

    if n_tokens <= 0 or n_points <= 0:
        return []
    grid = np.linspace(0, n_tokens - 1, n_points + 2)[1:-1]
    return [int(round(float(x))) for x in grid]


def select_readout_positions(
    n_tokens: int,
    boundaries: Sequence[int],
    n_chunk_latents: int = DEFAULT_N_CHUNK_LATENTS,
) -> ChunkReadout:
    if n_tokens <= 0:
        raise ValueError("select_readout_positions needs a non-empty trace.")
    if n_chunk_latents <= 0:
        return ChunkReadout(positions=[], boundaries=sorted(set(boundaries)))

    all_bounds = sorted({int(b) for b in boundaries if 0 <= int(b) < n_tokens})
    n_chunks = len(all_bounds) + (
        1 if all_bounds and all_bounds[-1] != n_tokens - 1 else 0
    )
    # Position 0 and the final position belong to the start/end latents.
    cands = [b for b in all_bounds if 0 < b < n_tokens - 1]

    if len(cands) >= n_chunk_latents:
        picks = np.linspace(0, len(cands) - 1, n_chunk_latents)
        positions = sorted({cands[int(round(float(p)))] for p in picks})
        n_filled = 0
    else:
        chosen = set(cands)
        for pos in _even_grid(n_tokens, n_chunk_latents):
            if len(chosen) >= n_chunk_latents:
                break
            chosen.add(pos)
        # Grid points can collide with boundaries; sweep remaining tokens.
        pos = 0
        while len(chosen) < n_chunk_latents and pos < n_tokens:
            chosen.add(pos)
            pos += 1
        positions = sorted(chosen)
        n_filled = len(positions) - len(set(cands) & set(positions))

    degenerate = len(positions) < n_chunk_latents
    if degenerate:
        positions = positions + [positions[-1]] * (n_chunk_latents - len(positions))

    return ChunkReadout(
        positions=positions,
        boundaries=all_bounds,
        n_chunks=n_chunks,
        n_filled=n_filled,
        degenerate=degenerate,
    )


def plan_chunk_readout(
    token_pieces: Sequence[str],
    n_chunk_latents: int = DEFAULT_N_CHUNK_LATENTS,
    delimiters: Sequence[str] = DEFAULT_SENTENCE_DELIMITERS,
) -> ChunkReadout:
    return select_readout_positions(
        n_tokens=len(token_pieces),
        boundaries=find_chunk_boundaries(token_pieces, delimiters=delimiters),
        n_chunk_latents=n_chunk_latents,
    )
