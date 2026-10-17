"""Workshop checks. Run after step 3:  python -m unittest discover -s tests -v"""
import math
import subprocess
import sys
import unittest

import torch

from core import settings
from core.dataset import get_batch, load_tokens
from nanogpt.model import GPT, GPTConfig
from core.vocabulary import build_vocabulary, decode, encode, load_vocabulary


class VocabularyTests(unittest.TestCase):
    def test_round_trip(self):
        vocabulary = build_vocabulary("ENGINEER: hello")
        self.assertEqual(decode(encode("ENGINEER", vocabulary), vocabulary), "ENGINEER")

    def test_unknown_character_fails_loudly(self):
        with self.assertRaises(KeyError):
            encode("z", build_vocabulary("abc"))


class PreparedDataTests(unittest.TestCase):
    def test_vocabulary_and_split(self):
        vocabulary = load_vocabulary(settings.VOCABULARY_FILE)
        train, validation = load_tokens(settings.TRAIN_TOKENS_FILE), load_tokens(settings.VALIDATION_TOKENS_FILE)
        self.assertEqual(len(vocabulary), 65)
        self.assertEqual(len(train) + len(validation), len(settings.RAW_TEXT_FILE.read_text(encoding="utf-8")))
        self.assertLess(int(max(train.max(), validation.max())), len(vocabulary))

    def test_batch_targets_are_inputs_shifted_by_one(self):
        x, y = get_batch(load_tokens(settings.TRAIN_TOKENS_FILE), 4, settings.CONTEXT_LENGTH)
        self.assertEqual(tuple(x.shape), (4, settings.CONTEXT_LENGTH))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))


class ModelTests(unittest.TestCase):
    def test_untrained_loss_is_near_random_guessing(self):
        torch.manual_seed(0)
        model = GPT(GPTConfig(block_size=settings.CONTEXT_LENGTH, vocab_size=65, n_layer=settings.LAYER_COUNT,
                              n_head=settings.HEAD_COUNT, n_embd=settings.EMBEDDING_SIZE, dropout=0.0, bias=False))
        x, y = get_batch(load_tokens(settings.TRAIN_TOKENS_FILE), 4, settings.CONTEXT_LENGTH)
        _, loss = model(x, y)
        self.assertAlmostEqual(loss.item(), math.log(65), delta=0.3)


class TrainedModelTests(unittest.TestCase):
    def test_checkpoint_learned_something(self):
        checkpoint = torch.load(settings.CHECKPOINT_FILE)
        self.assertLess(checkpoint["validation_loss"], 2.5)

    def test_step04_continues_the_prompt(self):
        result = subprocess.run(
            [sys.executable, "step04-generate-text.py", "--prompt", "ENGINEER:", "--characters", "50"],
            cwd=settings.WORKSHOP_DIR, capture_output=True, text=True, check=True)
        text = result.stdout.split("ENGINEER:", 1)[1]  # skip NanoGPT's "number of parameters" line
        self.assertGreaterEqual(len(text), 50)


if __name__ == "__main__":
    unittest.main()
