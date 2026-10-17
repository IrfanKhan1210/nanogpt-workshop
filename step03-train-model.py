"""Step 3 - Train the model on Shakespeare: guess -> measure error -> adjust weights -> repeat.

Run:  python step03-train-model.py      (about 30 s - 5 min on a CPU)
Makes: output/shakespeare.pt (the best model seen during training)

A simplified, readable version of NanoGPT's train.py (CPU only: no GPU, no wandb,
no mixed precision) with the same core ideas and default values.
Watch the loss fall: ~4.17 means random guessing, below 2.0 means it has learned spelling and style.
"""
import math
import time

import torch

from nanogpt.model import GPT, GPTConfig

from core import settings
from core.dataset import get_batch, load_tokens
from core.vocabulary import load_vocabulary


def learning_rate_at(step: int) -> float:
    """Warm up linearly, then slowly decrease along a cosine curve (NanoGPT's schedule)."""
    if step < settings.WARMUP_STEPS:  # phase 1: small steps while the weights are still random
        return settings.LEARNING_RATE * (step + 1) / (settings.WARMUP_STEPS + 1)
    progress = (step - settings.WARMUP_STEPS) / max(1, settings.MAX_STEPS - settings.WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))  # goes smoothly 1 -> 0
    return settings.MIN_LEARNING_RATE + cosine * (settings.LEARNING_RATE - settings.MIN_LEARNING_RATE)


@torch.no_grad()  # measuring only: no gradients needed, saves memory and time
def measure_loss(model, splits: dict) -> dict:
    """Average loss over a few random batches for each split (train / validation)."""
    model.eval()  # switch off training-only behaviour such as dropout
    results = {}
    for name, tokens in splits.items():
        losses = torch.zeros(settings.EVAL_BATCHES)
        for i in range(settings.EVAL_BATCHES):
            x, y = get_batch(tokens, settings.BATCH_SIZE, settings.CONTEXT_LENGTH)
            _, loss = model(x, y)  # NanoGPT returns (predictions, loss) when targets are given
            losses[i] = loss.item()
        results[name] = losses.mean().item()
    model.train()  # back to training mode
    return results


def save_checkpoint(model, step: int, validation_loss: float) -> None:
    """Save the weights plus the model shape, so step 4 can rebuild exactly this model."""
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),   # all learned numbers
        "model_config": vars(model.config),  # the shape: layers, heads, sizes
        "step": step,
        "validation_loss": validation_loss,
    }, settings.CHECKPOINT_FILE)


# 1. Load the encoded data from step 1 and build a fresh, random model.
torch.manual_seed(settings.RANDOM_SEED)  # repeatable runs
vocabulary = load_vocabulary(settings.VOCABULARY_FILE)
splits = {"train": load_tokens(settings.TRAIN_TOKENS_FILE),
          "validation": load_tokens(settings.VALIDATION_TOKENS_FILE)}
config = GPTConfig(
    block_size=settings.CONTEXT_LENGTH,  # how far back the model can look
    vocab_size=len(vocabulary),          # how many different characters it can output
    n_layer=settings.LAYER_COUNT,        # stacked Transformer blocks
    n_head=settings.HEAD_COUNT,          # attention heads per block
    n_embd=settings.EMBEDDING_SIZE,      # numbers per character inside the model
    dropout=settings.DROPOUT,
    bias=settings.USE_BIAS,
)
model = GPT(config)  # random weights: knows nothing yet

# 2. Create the optimizer: NanoGPT's AdamW setup (weight decay only on weight matrices).
optimizer = model.configure_optimizers(
    settings.WEIGHT_DECAY, settings.LEARNING_RATE, settings.ADAM_BETAS, "cpu")  # "cpu": NanoGPT asks the device type to pick its AdamW variant

# 3. The training loop.
best_validation_loss = float("inf")
started = time.time()
for step in range(settings.MAX_STEPS + 1):
    for group in optimizer.param_groups:  # apply this step's learning rate
        group["lr"] = learning_rate_at(step)

    # Every few hundred steps: measure progress and keep the best model so far.
    if step % settings.EVAL_EVERY_STEPS == 0 or step == settings.MAX_STEPS:
        losses = measure_loss(model, splits)
        print(f"step {step:5d} | train loss {losses['train']:.4f} | validation loss {losses['validation']:.4f}")
        if losses["validation"] < best_validation_loss:  # better on unseen text -> save it
            best_validation_loss = losses["validation"]
            save_checkpoint(model, step, best_validation_loss)
    if step == settings.MAX_STEPS:
        break

    x, y = get_batch(splits["train"], settings.BATCH_SIZE, settings.CONTEXT_LENGTH)
    _, loss = model(x, y)                   # a. guess the next characters and measure the error
    optimizer.zero_grad(set_to_none=True)   # b. forget the previous step's adjustments
    loss.backward()                         # c. work out how each weight contributed to the error
    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.GRADIENT_CLIP)  # d. cap extreme adjustments
    optimizer.step()                        # e. adjust every weight a little

    if step % settings.LOG_EVERY_STEPS == 0:
        print(f"step {step:5d} | batch loss {loss.item():.4f} | learning rate {learning_rate_at(step):.2e}")

# 4. Summary.
print(f"\nbest validation loss : {best_validation_loss:.4f}")
print(f"training time        : {time.time() - started:.1f} s")
print(f"saved model          : {settings.CHECKPOINT_FILE}")
