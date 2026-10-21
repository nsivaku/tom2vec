from tom2vec.chunking import (
    DEFAULT_K_LATENTS,
    DEFAULT_SENTENCE_DELIMITERS,
    ChunkReadout,
    find_chunk_boundaries,
    plan_chunk_readout,
    select_readout_positions,
)
from tom2vec.loss import tom_loss

__all__ = [
    "DEFAULT_K_LATENTS",
    "DEFAULT_SENTENCE_DELIMITERS",
    "ChunkReadout",
    "find_chunk_boundaries",
    "plan_chunk_readout",
    "select_readout_positions",
    "tom_loss",
    "ToM2Vec",
]


def __getattr__(name):
    # Lazy so the readout schedule can be imported without transformers.
    if name == "ToM2Vec":
        from tom2vec.model import ToM2Vec
        return ToM2Vec
    raise AttributeError(name)
