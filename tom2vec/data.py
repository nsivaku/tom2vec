from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from tom2vec.model import Messages


@dataclass
class Example:
    main: Messages
    partners: List[Messages]
    # (K, d) target
    target: Optional[torch.Tensor] = field(default=None, repr=False)


def load_pairs(path: str) -> List[Example]:
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                examples.append(Example(main=row["main"], partners=row["partners"]))
    if not examples:
        raise ValueError(f"no examples in {path}")
    return examples


def write_pairs(path: str, rows) -> int:
    """Write an iterable of {"main": ..., "partners": [...]} dicts as JSONL."""
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps({"main": row["main"], "partners": row["partners"]}) + "\n")
            n += 1
    return n
