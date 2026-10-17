"""Step 2 - Build an untrained GPT and look inside it before it learns anything.

Run:  python step02-build-model.py
Change the model shape in core/settings.py (LAYER_COUNT, EMBEDDING_SIZE ...) and rerun.
"""
import torch

from nanogpt.model import GPT, GPTConfig

from core import settings
from core.vocabulary import decode, encode, load_vocabulary

torch.manual_seed(settings.RANDOM_SEED)                 # same random weights every run
vocabulary = load_vocabulary(settings.VOCABULARY_FILE)  # made by step 1

# 1. Build the model: NanoGPT's GPT class filled with random numbers - it knows nothing yet.
config = GPTConfig(
    block_size=settings.CONTEXT_LENGTH,  # how far back the model can look
    vocab_size=len(vocabulary),          # how many different characters it can output
    n_layer=settings.LAYER_COUNT,        # stacked Transformer blocks
    n_head=settings.HEAD_COUNT,          # attention heads per block
    n_embd=settings.EMBEDDING_SIZE,      # numbers per character inside the model
    dropout=settings.DROPOUT,
    bias=settings.USE_BIAS,
)
model = GPT(config)
print(f"layers (Transformer blocks) : {config.n_layer}")
print(f"attention heads per layer   : {config.n_head}")
print(f"numbers per character       : {config.n_embd}")
print(f"context length              : {config.block_size} characters")
print(f"vocabulary size             : {config.vocab_size}")
print(f"trainable parameters        : {model.get_num_params(non_embedding=False):,}")

# 2. Where do the parameters live? Count them per part of the model.
print("\nparameters per part:")
for name, part in [("character embedding", model.transformer.wte),   # 65 x 128 lookup table
                   ("position embedding", model.transformer.wpe),    # 64 x 128 lookup table
                   ("one Transformer block", model.transformer.h[0]),
                   ("final LayerNorm", model.transformer.ln_f)]:
    print(f"  {name:<22}: {sum(p.numel() for p in part.parameters()):>8,}")

# 3. Measure how wrong the random model is: loss ~ ln(65) = 4.17 means pure guessing.
example = torch.tensor([encode("ENGINEER: I build", vocabulary)])
_, loss = model(example[:, :-1], example[:, 1:])  # predict each next character of the example
print(f"\nloss before training : {loss.item():.4f}  (random guessing = {torch.log(torch.tensor(65.0)).item():.4f})")

# 4. Let it write: before training it picks characters almost at random - compare with step 4.
prompt_ids = torch.tensor([encode(settings.DEFAULT_PROMPT, vocabulary)])
random_text = decode(model.generate(prompt_ids, 100)[0].tolist(), vocabulary)
print("\nuntrained model writes:\n" + random_text)
