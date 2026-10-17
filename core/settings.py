"""All workshop settings in one place.

Change a number here, rerun the steps, and watch what happens.
"""
from pathlib import Path

# ---- Folders -------------------------------------------------------------
WORKSHOP_DIR = Path(__file__).resolve().parent.parent  # the nanogpt-workshop/ folder
DATA_DIR = WORKSHOP_DIR / "data"                        # raw text + encoded data live here
OUTPUT_DIR = WORKSHOP_DIR / "output"                    # trained model is saved here

RAW_TEXT_FILE = DATA_DIR / "input.txt"            # tiny Shakespeare: ~1.1M characters of plays
VOCABULARY_FILE = DATA_DIR / "vocabulary.json"    # character <-> number lookup table
TRAIN_TOKENS_FILE = DATA_DIR / "train.bin"        # 90% of the text, as numbers, for learning
VALIDATION_TOKENS_FILE = DATA_DIR / "val.bin"     # 10% held back, to check the model is not memorising
CHECKPOINT_FILE = OUTPUT_DIR / "shakespeare.pt"   # the trained model weights

# ---- Data ----------------------------------------------------------------
TRAIN_FRACTION = 0.9  # first 90% of the text trains the model, last 10% tests it

# ---- Model shape (how big the "brain" is) --------------------------------
CONTEXT_LENGTH = 64   # how many characters the model can look back at (NanoGPT: block_size)
LAYER_COUNT = 4       # how many Transformer blocks are stacked (n_layer)
HEAD_COUNT = 4        # attention heads per block (n_head)
EMBEDDING_SIZE = 128  # how many numbers describe each character inside the model (n_embd)
DROPOUT = 0.0         # randomly switch off neurons while training; 0 = off (small model, short run)
USE_BIAS = False      # NanoGPT default: no bias terms in Linear/LayerNorm layers

# ---- Training (how the "brain" learns; everything runs on the CPU) -------
RANDOM_SEED = 1337          # same seed -> same random choices -> repeatable results
BATCH_SIZE = 12             # how many 64-character snippets are learned from at once
MAX_STEPS = 2000            # how many learning steps to run in total
LEARNING_RATE = 1e-3        # how big each weight adjustment is at its largest
MIN_LEARNING_RATE = 1e-4    # the smallest adjustment size, reached at the end of training
WARMUP_STEPS = 100          # start with tiny adjustments, grow to LEARNING_RATE over these steps
WEIGHT_DECAY = 0.1          # gently pulls weights towards zero so none grow too large
ADAM_BETAS = (0.9, 0.99)    # AdamW's memory of past adjustments (NanoGPT Shakespeare values)
GRADIENT_CLIP = 1.0         # caps any single adjustment so one bad batch cannot wreck the model
EVAL_EVERY_STEPS = 250      # how often to measure train/validation loss
EVAL_BATCHES = 20           # how many batches to average when measuring loss
LOG_EVERY_STEPS = 50        # how often to print progress

# ---- Generation (how the trained model writes) ---------------------------
DEFAULT_PROMPT = "ENGINEER:\n"  # text the model continues from
NEW_CHARACTER_COUNT = 300       # how many characters to write
TEMPERATURE = 0.8               # <1 = safer, more repetitive; >1 = wilder, more mistakes
TOP_K = 20                      # only choose among the 20 most likely next characters
