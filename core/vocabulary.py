"""Turn characters into numbers and back.

A neural network only understands numbers, so every character gets an ID:
'a' -> 39, 'b' -> 40, ... The list of all characters is the "vocabulary".
"""
import json
from pathlib import Path


def build_vocabulary(text: str) -> list[str]:
    """Return every distinct character in the text, sorted so the order is repeatable."""
    return sorted(set(text))  # set() removes duplicates; sorted() fixes the order


def encode(text: str, vocabulary: list[str]) -> list[int]:
    """Text -> list of character IDs, e.g. 'hi' -> [46, 47]."""
    char_to_id = {char: index for index, char in enumerate(vocabulary)}  # lookup table
    return [char_to_id[char] for char in text]  # KeyError here = character not in vocabulary


def decode(token_ids: list[int], vocabulary: list[str]) -> str:
    """List of character IDs -> text, e.g. [46, 47] -> 'hi'."""
    return "".join(vocabulary[token_id] for token_id in token_ids)


def save_vocabulary(vocabulary: list[str], path: Path) -> None:
    """Store the vocabulary as JSON so training and generation use the exact same IDs."""
    path.write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")


def load_vocabulary(path: Path) -> list[str]:
    """Read the vocabulary saved by step 1. Fails loudly if step 1 has not run."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run step01-prepare-data.py first")
    return json.loads(path.read_text(encoding="utf-8"))
