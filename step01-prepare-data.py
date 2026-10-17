"""Step 1 - Turn Shakespeare's text into numbers the model can learn from.

Run:  python step01-prepare-data.py
Makes: data/vocabulary.json, data/train.bin, data/val.bin
"""
import numpy as np

from core import settings
from core.dataset import TOKEN_DTYPE
from core.vocabulary import build_vocabulary, encode, save_vocabulary

# 1. Read the raw text: ~1.1M characters of Shakespeare plays.
text = settings.RAW_TEXT_FILE.read_text(encoding="utf-8")
print(f"characters in the text : {len(text):,}")
print(f"first 100 characters   : {text[:100]!r}")

# 2. Build the vocabulary: every distinct character gets a number (its position in the list).
vocabulary = build_vocabulary(text)
save_vocabulary(vocabulary, settings.VOCABULARY_FILE)  # steps 2-4 must use the exact same numbers
print(f"vocabulary size        : {len(vocabulary)}")
print(f"vocabulary             : {''.join(vocabulary)!r}")
print(f"'ENGINEER' as numbers  : {encode('ENGINEER', vocabulary)}")

# 3. Split: the first 90% teaches the model, the last 10% is held back to check it is not memorising.
split_at = int(len(text) * settings.TRAIN_FRACTION)  # character position of the 90% mark
train_ids = np.array(encode(text[:split_at], vocabulary), dtype=TOKEN_DTYPE)
validation_ids = np.array(encode(text[split_at:], vocabulary), dtype=TOKEN_DTYPE)

# 4. Save both parts as raw binary numbers: compact (2 bytes each) and fast to read during training.
train_ids.tofile(settings.TRAIN_TOKENS_FILE)
validation_ids.tofile(settings.VALIDATION_TOKENS_FILE)
print(f"training tokens        : {len(train_ids):,}  -> {settings.TRAIN_TOKENS_FILE.name}")
print(f"validation tokens      : {len(validation_ids):,}  -> {settings.VALIDATION_TOKENS_FILE.name}")
