# NanoGPT Workshop: Build Your Own Shakespeare Model

Train a small GPT, from scratch on a laptop CPU, that writes text in the style of Shakespeare.
The model is Andrej Karpathy's NanoGPT (`nanogpt/model.py`, unchanged). Around it sit four short,
commented step scripts that walk through building the model one stage at a time.

## What It Does

- Turns 1.1 million characters of Shakespeare into numbers a neural network can read.
- Builds a 4-layer GPT (about 800,000 learnable numbers) and shows what is inside it.
- Trains it in about 30 seconds to a few minutes on a CPU. No GPU is needed.
- Writes new Shakespeare-style text from any opening line you give it.

## Folder Layout

```text
nanogpt-workshop/
├── step01-prepare-data.py    # Step 1: text -> numbers -> train/validation files
├── step02-build-model.py     # Step 2: build an untrained GPT and look inside it
├── step03-train-model.py     # Step 3: the training loop
├── step04-generate-text.py   # Step 4: let the trained model write
├── core/                     # small helpers shared by the steps
│   ├── settings.py           #   every number you can change, in one place
│   ├── vocabulary.py         #   character <-> number conversion
│   └── dataset.py            #   read the number files, pick random training snippets
├── nanogpt/model.py          # NanoGPT's original GPT model (reference, do not edit)
├── data/input.txt            # the Tiny Shakespeare text
├── tests/test_workshop.py    # checks that every step worked
└── output/                   # created by step 3: the trained model
```

## How It Works

A GPT does one thing: it looks at the text so far and guesses the next character.
"ENGINEE" -> "R". Writing a whole paragraph is that guess repeated, one character at a time.

To learn, the model reads millions of real snippets of Shakespeare. Each time it guesses
the next character, it measures how wrong it was (the loss) and slightly adjusts its 800,000
internal numbers (the weights) to be less wrong next time. After 2,000 rounds of this, it has
learned spelling, line breaks, character names and the rhythm of the plays.

```text
input.txt ──step 1──> numbers ──step 2──> random GPT ──step 3──> trained GPT ──step 4──> new text
```

## Setup

Requires Python 3.14. Run every command from inside this folder.

```sh
cd nanogpt-workshop
uv venv && source .venv/bin/activate     # or: python -m venv .venv && source .venv/bin/activate
uv pip install -r requirements-linux-cpu.txt --index-strategy unsafe-best-match   # Linux
# uv pip install -r requirements.txt                                             # macOS / Windows
```

The only libraries are NumPy and PyTorch (CPU build).

## Step-by-Step Guide

### Step 1: Prepare the data

```sh
python step01-prepare-data.py
```

What happens:
1. Reads `data/input.txt` (1,115,394 characters).
2. Finds every distinct character and gives each one a number. This list is the vocabulary:
   65 characters (letters, digits, punctuation, space, newline).
3. Converts the whole text to numbers, e.g. `ENGINEER` -> `[17, 26, 19, 21, 26, 17, 17, 30]`.
4. Splits it: the first 90% for learning, the last 10% held back to test the model on text it never trained on.
5. Saves `data/vocabulary.json`, `data/train.bin` and `data/val.bin`.

Expected: `training tokens: 1,003,854` and `validation tokens: 111,540`.

### Step 2: Build the model

```sh
python step02-build-model.py
```

What happens:
1. Creates NanoGPT's `GPTConfig` from the numbers in `core/settings.py`, then builds a `GPT` with random weights.
2. Prints its shape: 4 layers, 4 attention heads, 128 numbers per character, a 64-character context.
3. Shows where the 804,096 parameters live: character table, position table, Transformer blocks.
4. Measures the loss before training: about 4.17. That is the score for guessing at random from 65 characters.
5. Lets the untrained model write: pure gibberish, such as `oONq!ds3:Ew z!!SuVOt...`.

The main parts of the model:

| Part | What it does in plain words |
|------|-----------------------------|
| Character embedding | Looks up 128 numbers that describe each character |
| Position embedding | Adds 128 numbers that describe where in the line the character sits |
| Attention (in each block) | Each character looks back at earlier characters and picks the useful ones |
| Feed-forward (in each block) | A small neural network that combines what attention found |
| 4 stacked blocks | Repeats attention + feed-forward to build deeper understanding |
| Output head | Turns the result into a score for each of the 65 possible next characters |

To read how each part is coded, open `nanogpt/model.py`. The class names match the table:
`CausalSelfAttention`, `MLP`, `Block`, `GPT`.

### Step 3: Train the model

```sh
python step03-train-model.py
```

What happens, 2,000 times:
1. Pick 12 random 64-character snippets (`get_batch`). The answer for each snippet is the same text shifted one character to the right.
2. The model guesses the next character at every position and computes the loss.
3. `loss.backward()` works out how each weight contributed to the error.
4. The optimizer (AdamW) nudges every weight a little. The size of each nudge (the learning rate) starts small, rises, then slowly shrinks.
5. Every 250 steps it measures loss on both training and validation text, and saves the best model to `output/shakespeare.pt`.

Expected output (on our test run, 30 seconds on CPU):

```text
step     0 | train loss 4.1686 | validation loss 4.1659
step  1000 | train loss 1.9693 | validation loss 2.0513
step  2000 | train loss 1.7609 | validation loss 1.8843
```

If the validation loss falls along with the training loss, the model is learning. If only the
training loss falls, it is memorising.

### Step 4: Generate text

```sh
python step04-generate-text.py
python step04-generate-text.py --prompt "KING:" --characters 200 --temperature 1.0 --seed 7
```

| Option | Meaning |
|--------|---------|
| `--prompt` | The opening text the model continues. Type `\n` for a new line. Default `ENGINEER:\n` |
| `--characters` | How many characters to write. Default 300 |
| `--temperature` | Below 1 = safer and more repetitive; above 1 = more adventurous, more mistakes. Default 0.8 |
| `--seed` | Change it to get a different sample from the same prompt |

Sample output:

```text
ENGINEER:
I by dother love doth not shelf appud and thee,
And and and sorbe sorrowit fring to him.
```

It looks like Shakespeare (names in capitals, verse-like lines, "thee" and "doth") but the
meaning is weak. That is expected from a model this small and trained this briefly.

## Experiments to Try

Change a value in `core/settings.py`, then rerun step 3 and step 4 and compare.

| Change | Question to explore |
|--------|---------------------|
| `MAX_STEPS = 5000` | Does longer training give better text? Does validation loss keep falling? |
| `LAYER_COUNT = 6`, `EMBEDDING_SIZE = 192` | Does a bigger model write better? How much slower is training? |
| `CONTEXT_LENGTH = 128` | Does seeing more past text help the model stay consistent? |
| `LEARNING_RATE = 1e-2` | What happens when each adjustment is too large? |
| `TEMPERATURE = 0.3` or `1.5` | How does the writing change, with no retraining? |
| Replace `data/input.txt` with your own text | Can the model learn a different style? Rerun all steps |

Step 4 rebuilds the model with the shape saved in `output/shakespeare.pt`, so a change to the
model shape only takes effect after you rerun step 3.

## Test

Run after step 3:

```sh
python -m unittest discover -s tests -v
```

The 7 tests check the vocabulary, the data split, the snippet shifting, the untrained loss,
the saved model, and that step 4 writes text. Expected: `Ran 7 tests ... OK`.

## Troubleshooting

| Message | Fix |
|---------|-----|
| `vocabulary.json not found - run step01...` | Run step 1 first |
| `shakespeare.pt not found - run step03...` | Run step 3 first |
| `KeyError` in step 4 | Your prompt has a character that is not in the vocabulary (e.g. `é`, a digit other than 3). Use plain English letters and punctuation |
| `IndexError` or garbled text after changing `input.txt` | Steps 1 and 3 must use the same vocabulary. Rerun step 1, then step 3 |
| `ModuleNotFoundError: core` | Run the commands from inside the `nanogpt-workshop/` folder |

## License

`nanogpt/model.py` is from [NanoGPT](https://github.com/karpathy/nanoGPT) (commit `3adf61e`),
MIT licence: see `nanogpt/LICENSE`. `data/input.txt` is the public Tiny Shakespeare dataset.
