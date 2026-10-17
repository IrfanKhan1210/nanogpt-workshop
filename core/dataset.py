"""Shared data helpers used by several steps: read encoded token files and serve training snippets."""
from pathlib import Path

import numpy as np
import torch

# uint16 holds IDs 0..65535 in 2 bytes each - plenty for a 65-character vocabulary
TOKEN_DTYPE = np.uint16


def load_tokens(path: Path) -> np.ndarray:
    """Open an encoded .bin file without loading it all into memory (memory-mapped)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run step01-prepare-data.py first")
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def get_batch(tokens: np.ndarray, batch_size: int, context_length: int):
    """Pick random snippets: X is the input, Y is the same snippet shifted by one character.

    Example with context_length=5 and text "ENGINEER":
        X = "ENGIN"   Y = "NGINE"
    At every position the model must predict the next character: E->N, N->G, G->I ...
    """
    # random start positions, leaving room for context_length + 1 characters
    starts = torch.randint(len(tokens) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(tokens[s:s + context_length].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(tokens[s + 1:s + 1 + context_length].astype(np.int64)) for s in starts])
    return x, y  # shape of each: [batch_size, context_length]
