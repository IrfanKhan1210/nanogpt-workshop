"""Step 4 - Let the trained model write Shakespeare-style text.

Run:  python step04-generate-text.py
      python step04-generate-text.py --prompt "KING:" --characters 200 --temperature 1.0 --seed 7
"""
import argparse

import torch

from nanogpt.model import GPT, GPTConfig

from core import settings
from core.vocabulary import decode, encode, load_vocabulary

# 1. Read the options: what to continue, how long, how adventurous.
parser = argparse.ArgumentParser(description="Continue a prompt in Shakespeare style.")
parser.add_argument("--prompt", default=settings.DEFAULT_PROMPT, help="text to continue")
parser.add_argument("--characters", type=int, default=settings.NEW_CHARACTER_COUNT, help="how many characters to write")
parser.add_argument("--temperature", type=float, default=settings.TEMPERATURE, help="<1 safer, >1 wilder")
parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED, help="change for a different sample")
args = parser.parse_args()
prompt = args.prompt.replace("\\n", "\n")  # allow typing \n on the command line
torch.manual_seed(args.seed)               # same seed -> same text

# 2. Rebuild the trained model: same shape as in training, then put the learned numbers back.
if not settings.CHECKPOINT_FILE.exists():
    raise FileNotFoundError(f"{settings.CHECKPOINT_FILE} not found - run step03-train-model.py first")
checkpoint = torch.load(settings.CHECKPOINT_FILE)
model = GPT(GPTConfig(**checkpoint["model_config"]))
model.load_state_dict(checkpoint["model_state"])
model.eval()                                # generation mode, not training
vocabulary = load_vocabulary(settings.VOCABULARY_FILE)

# 3. Write: NanoGPT's generate() predicts the next character, samples one, appends it, repeats.
prompt_ids = torch.tensor([encode(prompt, vocabulary)], dtype=torch.long)
with torch.no_grad():                       # writing only, no learning
    output_ids = model.generate(prompt_ids, args.characters, temperature=args.temperature, top_k=settings.TOP_K)
print(decode(output_ids[0].tolist(), vocabulary))
