import unittest
import io
import contextlib
import math
from bpe import Token
from bpe import TokenFreq
from bpe import TokenFreqSet
from bpe import Vocabulary
from embed import Embed
from embed import TrainingOptions
from embed import InitResult
from embed import EpochResult

# One Embed shared by all the tests below: creating it trains the BPE tokenizer on the
# corpus, which is slow and prints a lot, so it is done once and its output hidden.
EMBED: Embed


def setUpModule():
    global EMBED
    with contextlib.redirect_stdout(io.StringIO()):
        EMBED = Embed()


class TestBuildTrainingPairs(unittest.TestCase):
    def test_window_of_one(self):
        pairs = EMBED.build_training_pairs([[9, 38, 249]], 1)
        self.assertEqual([(9, 38), (38, 9), (38, 249), (249, 38)], pairs)

    def test_window_of_two(self):
        pairs = EMBED.build_training_pairs([[1, 2, 3, 4]], 2)
        self.assertEqual([
            (1, 2), (1, 3),
            (2, 1), (2, 3), (2, 4),
            (3, 1), (3, 2), (3, 4),
            (4, 2), (4, 3),
        ], pairs)

    def test_window_larger_than_sentence(self):
        pairs = EMBED.build_training_pairs([[1, 2]], 5)
        self.assertEqual([(1, 2), (2, 1)], pairs)

    def test_pairs_do_not_cross_sentences(self):
        pairs = EMBED.build_training_pairs([[1, 2], [3, 4]], 2)
        self.assertEqual([(1, 2), (2, 1), (3, 4), (4, 3)], pairs)

    def test_repeated_tokens_are_kept(self):
        pairs = EMBED.build_training_pairs([[5, 5]], 1)
        self.assertEqual([(5, 5), (5, 5)], pairs)

    def test_empty_and_single_token_sentences(self):
        self.assertEqual([], EMBED.build_training_pairs([[], [7]], 2))


class TestBuildNegativeSamplingTable(unittest.TestCase):
    def _vocabulary(self, token_string_to_freq: dict[str, int]) -> Vocabulary:
        token_freqs = TokenFreqSet()
        for token_string, freq in token_string_to_freq.items():
            token_freqs.add(TokenFreq(Token(token_string), freq))
        return Vocabulary(token_freqs)

    def test_three_token_example(self):
        # The worked example from the comment: the:100, cat:10, mat:1
        cumulative = EMBED.build_negative_sampling_table(self._vocabulary({"the": 100, "cat": 10, "mat": 1}))
        self.assertEqual(3, len(cumulative))
        self.assertAlmostEqual(0.827, cumulative[0], places=3)
        self.assertAlmostEqual(0.974, cumulative[1], places=3)
        self.assertAlmostEqual(1.0, cumulative[2])

    def test_equal_frequencies_give_equal_shares(self):
        cumulative = EMBED.build_negative_sampling_table(self._vocabulary({"a": 3, "b": 3, "c": 3, "d": 3}))
        for expected, actual in zip([0.25, 0.5, 0.75, 1.0], cumulative):
            self.assertAlmostEqual(expected, actual)

    def test_cumulative_never_decreases(self):
        cumulative = EMBED.build_negative_sampling_table(self._vocabulary({"a": 7, "b": 1, "c": 50, "d": 2}))
        for i in range(len(cumulative) - 1):
            self.assertLessEqual(cumulative[i], cumulative[i + 1])


class TestSampleNegative(unittest.TestCase):
    def _fixed_rand(self, values: list[float]):
        values_iter = iter(values)
        return lambda: next(values_iter)

    def test_examples_from_comment(self):
        cumulative = [0.827, 0.974, 1.0]
        rand = self._fixed_rand([0.50, 0.90, 0.99])
        self.assertEqual([0, 1, 2], [EMBED.sample_negative(cumulative, rand) for _ in range(3)])

    def test_value_on_a_boundary_picks_that_token(self):
        cumulative = [0.25, 0.5, 0.75, 1.0]
        rand = self._fixed_rand([0.0, 0.25, 0.5, 0.75])
        self.assertEqual([0, 0, 1, 2], [EMBED.sample_negative(cumulative, rand) for _ in range(4)])

    def test_never_returns_past_the_last_index(self):
        # Rounding can leave the last entry just below 1
        cumulative = [0.5, 0.9999999]
        rand = self._fixed_rand([0.99999999])
        self.assertEqual(1, EMBED.sample_negative(cumulative, rand))

    def test_matches_typescript_for_seed_42(self):
        # Reference values from running the TypeScript sampleNegative with EMBED.mulberry32(42)
        rand = EMBED.mulberry32(42)
        samples = [EMBED.sample_negative([0.827, 0.974, 1.0], rand) for _ in range(10)]
        self.assertEqual([0, 0, 1, 0, 0, 0, 0, 0, 1, 0], samples)


class TestInitWeights(unittest.TestCase):
    def test_sizes(self):
        w_in, w_out = EMBED.init_weights(5, 3, EMBED.mulberry32(1))
        self.assertEqual(15, len(w_in))
        self.assertEqual(15, len(w_out))

    def test_values_within_scale(self):
        dim = 4
        half_scale = 0.5 / dim / 2
        w_in, w_out = EMBED.init_weights(10, dim, EMBED.mulberry32(7))
        for weight in w_in + w_out:
            self.assertLessEqual(abs(weight), half_scale)

    def test_matches_typescript_for_seed_42(self):
        # Reference values from running the TypeScript initialisation with EMBED.mulberry32(42),
        # vocabSize = 2, dim = 2
        w_in, w_out = EMBED.init_weights(2, 2, EMBED.mulberry32(42))
        self.assertEqual([0.025275937980040908, 0.08811644837260246, -0.08129652531351894, -0.056693001417443156], w_in)
        self.assertEqual([-0.012927360250614583, 0.04243351035984233, 0.006648135546129197, 0.031186163483653218], w_out)


class TestShuffle(unittest.TestCase):
    def test_keeps_the_same_items(self):
        pairs = [(i, i + 1) for i in range(20)]
        shuffled = list(pairs)
        EMBED.shuffle(shuffled, EMBED.mulberry32(3))
        self.assertEqual(sorted(pairs), sorted(shuffled))
        self.assertNotEqual(pairs, shuffled)

    def test_same_seed_gives_same_order(self):
        first = [(i, i) for i in range(10)]
        second = [(i, i) for i in range(10)]
        EMBED.shuffle(first, EMBED.mulberry32(42))
        EMBED.shuffle(second, EMBED.mulberry32(42))
        self.assertEqual(first, second)

    def test_matches_typescript_for_seed_42(self):
        # Reference order from running the TypeScript Fisher-Yates shuffle on 0..9 with EMBED.mulberry32(42)
        items = [(i, i) for i in range(10)]
        EMBED.shuffle(items, EMBED.mulberry32(42))
        self.assertEqual([0, 7, 3, 5, 2, 1, 8, 9, 4, 6], [item[0] for item in items])

    def test_empty_and_single_item(self):
        empty: list[tuple[int, int]] = []
        EMBED.shuffle(empty, EMBED.mulberry32(1))
        self.assertEqual([], empty)
        single = [(1, 2)]
        EMBED.shuffle(single, EMBED.mulberry32(1))
        self.assertEqual([(1, 2)], single)


class TestLearningRate(unittest.TestCase):
    def test_falls_in_a_straight_line(self):
        rates = [EMBED.learning_rate(epoch, 4, 0.025, 0.001) for epoch in range(5)]
        for expected, actual in zip([0.025, 0.019, 0.013, 0.007, 0.001], rates):
            self.assertAlmostEqual(expected, actual)

    def test_zero_epochs_uses_start_rate(self):
        self.assertEqual(0.025, EMBED.learning_rate(0, 0, 0.025, 0.001))


class TestTrainPair(unittest.TestCase):
    def test_positive_example_from_comment(self):
        w_in = [0.5, 0.5]
        w_out = [0.5, -0.5]
        loss = EMBED.train_pair(w_in, w_out, 2, 0, 0, 1, 0.1)
        for expected, actual in zip([0.525, 0.475], w_in):
            self.assertAlmostEqual(expected, actual)
        for expected, actual in zip([0.525, -0.475], w_out):
            self.assertAlmostEqual(expected, actual)
        self.assertAlmostEqual(math.log(2), loss)

    def test_positive_pair_score_goes_up(self):
        # Rows: target is token 0 in w_in, other is token 1 in w_out
        w_in = [0.1, 0.2, 0.0, 0.0]
        w_out = [0.0, 0.0, 0.3, -0.1]
        before = self._dot(w_in, w_out, 0, 1)
        EMBED.train_pair(w_in, w_out, 2, 0, 1, 1, 0.5)
        self.assertGreater(self._dot(w_in, w_out, 0, 1), before)

    def test_negative_pair_score_goes_down(self):
        w_in = [0.1, 0.2, 0.0, 0.0]
        w_out = [0.0, 0.0, 0.3, -0.1]
        before = self._dot(w_in, w_out, 0, 1)
        EMBED.train_pair(w_in, w_out, 2, 0, 1, 0, 0.5)
        self.assertLess(self._dot(w_in, w_out, 0, 1), before)

    def test_only_the_two_rows_change(self):
        w_in = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        w_out = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        EMBED.train_pair(w_in, w_out, 2, 0, 2, 1, 0.5)
        self.assertEqual([0.3, 0.4, 0.5, 0.6], w_in[2:])
        self.assertEqual([0.6, 0.5, 0.4, 0.3], w_out[:4])

    def test_loss_for_negative_pair(self):
        # dot = 0 -> score 0.5 -> loss = -log(1 - 0.5)
        self.assertAlmostEqual(math.log(2), EMBED.train_pair([0.0, 0.0], [0.0, 0.0], 2, 0, 0, 0, 0.1))

    def _dot(self, w_in, w_out, target, other, dim=2):
        return sum(w_in[target * dim + d] * w_out[other * dim + d] for d in range(dim))


class TestTrainEpoch(unittest.TestCase):
    def test_no_pairs_gives_zero_loss(self):
        self.assertEqual(0.0, EMBED.train_epoch([], [0.0], [0.0], 1, 3, [1.0], EMBED.mulberry32(1), 0.1))

    def test_loss_falls_over_epochs(self):
        rand = EMBED.mulberry32(42)
        w_in = [(rand() - 0.5) * 0.1 for _ in range(4 * 2)]
        w_out = [(rand() - 0.5) * 0.1 for _ in range(4 * 2)]
        pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
        cumulative = [0.25, 0.5, 0.75, 1.0]
        losses = [EMBED.train_epoch(pairs, w_in, w_out, 2, 2, cumulative, rand, 0.5) for _ in range(30)]
        self.assertLess(losses[-1], losses[0])

    def test_negative_equal_to_context_is_skipped(self):
        # Only one token, so every negative is the context itself: only the positive is trained
        w_in = [0.0, 0.0]
        w_out = [0.0, 0.0]
        loss = EMBED.train_epoch([(0, 0)], w_in, w_out, 2, 5, [1.0], EMBED.mulberry32(1), 0.1)
        self.assertAlmostEqual(math.log(2), loss)


class TestRound6(unittest.TestCase):
    def test_rounds_to_six_places(self):
        self.assertEqual(2.130895, EMBED.round_6(2.1308954))
        self.assertEqual(2.130896, EMBED.round_6(2.1308956))

    def test_halves_round_up_like_javascript(self):
        # Python's round(0.5) is 0 (round half to even); Math.round(0.5) is 1
        self.assertEqual(0.000001, EMBED.round_6(0.0000005))


class TestTrainSkipGram(unittest.TestCase):
    def test_yields_init_then_epoch_results(self):
        with contextlib.redirect_stdout(io.StringIO()):
            results = list(Embed().train_skip_gram(TrainingOptions([], 3, 4, 2, 2)))
        self.assertIsInstance(results[0], InitResult)
        self.assertEqual(4, results[0].embeddingDim)
        self.assertEqual([0, 1, 2, 3], [result.epoch for result in results[1:]])
        for result in results[1:]:
            self.assertIsInstance(result, EpochResult)
        self.assertLess(results[-1].loss, results[1].loss)

    def test_matches_typescript_losses(self):
        # Reference losses from running the TypeScript training loop on the same pairs
        # and sampling table, 10 epochs, dim 8, window 2, 3 negative samples
        with contextlib.redirect_stdout(io.StringIO()):
            results = list(Embed().train_skip_gram(TrainingOptions([], 10, 8, 2, 3)))
        self.assertEqual(
            [2.75102, 2.748616, 2.709061, 2.584951, 2.433389, 2.324782, 2.24623, 2.192255, 2.158386, 2.149308, 2.130895],
            [result.loss for result in results[1:]])


if __name__ == "__main__":
    unittest.main()
