import unittest
import math
from bpe import Token, TokenFreq, TokenFreqSet, Vocabulary
from random_numbers import Mulberry32
from skip_gram_network import SkipGramNetwork
from skip_gram_trainer import SkipGramTrainer


class TestBuildNegativeSamplingTable(unittest.TestCase):
    def _vocabulary(self, token_string_to_freq: dict[str, int]) -> Vocabulary:
        token_freqs = TokenFreqSet()
        for token_string, freq in token_string_to_freq.items():
            token_freqs.add(TokenFreq(Token(token_string), freq))
        return Vocabulary(token_freqs)

    def test_three_token_example(self):
        # The worked example from the comment: the:100, cat:10, mat:1
        cumulative = SkipGramTrainer.build_negative_sampling_table(self._vocabulary({"the": 100, "cat": 10, "mat": 1}))
        self.assertEqual(3, len(cumulative))
        self.assertAlmostEqual(0.827, cumulative[0], places=3)
        self.assertAlmostEqual(0.974, cumulative[1], places=3)
        self.assertAlmostEqual(1.0, cumulative[2])

    def test_equal_frequencies_give_equal_shares(self):
        cumulative = SkipGramTrainer.build_negative_sampling_table(self._vocabulary({"a": 3, "b": 3, "c": 3, "d": 3}))
        for expected, actual in zip([0.25, 0.5, 0.75, 1.0], cumulative):
            self.assertAlmostEqual(expected, actual)

    def test_cumulative_never_decreases(self):
        cumulative = SkipGramTrainer.build_negative_sampling_table(self._vocabulary({"a": 7, "b": 1, "c": 50, "d": 2}))
        for i in range(len(cumulative) - 1):
            self.assertLessEqual(cumulative[i], cumulative[i + 1])


class TestSampleNegative(unittest.TestCase):
    def _fixed_rand(self, values: list[float]):
        values_iter = iter(values)
        return lambda: next(values_iter)

    def test_examples_from_comment(self):
        cumulative = [0.827, 0.974, 1.0]
        rand = self._fixed_rand([0.50, 0.90, 0.99])
        self.assertEqual([0, 1, 2], [SkipGramTrainer.sample_negative(cumulative, rand) for _ in range(3)])

    def test_value_on_a_boundary_picks_that_token(self):
        cumulative = [0.25, 0.5, 0.75, 1.0]
        rand = self._fixed_rand([0.0, 0.25, 0.5, 0.75])
        self.assertEqual([0, 0, 1, 2], [SkipGramTrainer.sample_negative(cumulative, rand) for _ in range(4)])

    def test_never_returns_past_the_last_index(self):
        # Rounding can leave the last entry just below 1
        cumulative = [0.5, 0.9999999]
        rand = self._fixed_rand([0.99999999])
        self.assertEqual(1, SkipGramTrainer.sample_negative(cumulative, rand))

    def test_matches_typescript_for_seed_42(self):
        # Reference values from running the TypeScript sampleNegative with Mulberry32(42)
        rand = Mulberry32(42)
        samples = [SkipGramTrainer.sample_negative([0.827, 0.974, 1.0], rand) for _ in range(10)]
        self.assertEqual([0, 0, 1, 0, 0, 0, 0, 0, 1, 0], samples)


class TestShuffle(unittest.TestCase):
    def test_keeps_the_same_items(self):
        pairs = [(i, i + 1) for i in range(20)]
        shuffled = list(pairs)
        SkipGramTrainer.shuffle(shuffled, Mulberry32(3))
        self.assertEqual(sorted(pairs), sorted(shuffled))
        self.assertNotEqual(pairs, shuffled)

    def test_same_seed_gives_same_order(self):
        first = [(i, i) for i in range(10)]
        second = [(i, i) for i in range(10)]
        SkipGramTrainer.shuffle(first, Mulberry32(42))
        SkipGramTrainer.shuffle(second, Mulberry32(42))
        self.assertEqual(first, second)

    def test_matches_typescript_for_seed_42(self):
        # Reference order from running the TypeScript Fisher-Yates shuffle on 0..9 with Mulberry32(42)
        items = [(i, i) for i in range(10)]
        SkipGramTrainer.shuffle(items, Mulberry32(42))
        self.assertEqual([0, 7, 3, 5, 2, 1, 8, 9, 4, 6], [item[0] for item in items])

    def test_empty_and_single_item(self):
        empty: list[tuple[int, int]] = []
        SkipGramTrainer.shuffle(empty, Mulberry32(1))
        self.assertEqual([], empty)
        single = [(1, 2)]
        SkipGramTrainer.shuffle(single, Mulberry32(1))
        self.assertEqual([(1, 2)], single)


class TestLearningRate(unittest.TestCase):
    def test_falls_in_a_straight_line(self):
        rates = [SkipGramTrainer.learning_rate(epoch, 4, 0.025, 0.001) for epoch in range(5)]
        for expected, actual in zip([0.025, 0.019, 0.013, 0.007, 0.001], rates):
            self.assertAlmostEqual(expected, actual)

    def test_zero_epochs_uses_start_rate(self):
        self.assertEqual(0.025, SkipGramTrainer.learning_rate(0, 0, 0.025, 0.001))


class TestTrainEpoch(unittest.TestCase):
    def test_no_pairs_gives_zero_loss(self):
        network = SkipGramNetwork(["a"], 1)
        self.assertEqual(0.0, SkipGramTrainer(network, [], [1.0], Mulberry32(1), 3).train_epoch(0.1))

    def test_loss_falls_over_epochs(self):
        rand = Mulberry32(42)
        network = SkipGramNetwork(["a", "b", "c", "d"], 2)
        for token in range(4):
            network.set_embedding(token, [(rand() - 0.5) * 0.1 for _ in range(2)])
        for token in range(4):
            network.set_context_vector(token, [(rand() - 0.5) * 0.1 for _ in range(2)])
        pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
        cumulative = [0.25, 0.5, 0.75, 1.0]
        trainer = SkipGramTrainer(network, pairs, cumulative, rand, 2)
        losses = [trainer.train_epoch(0.5) for _ in range(30)]
        self.assertLess(losses[-1], losses[0])

    def test_negative_equal_to_context_is_skipped(self):
        # Only one token, so every negative is the context itself: only the positive is trained
        network = SkipGramNetwork(["a"], 2)
        loss = SkipGramTrainer(network, [(0, 0)], [1.0], Mulberry32(1), 5).train_epoch(0.1)
        self.assertAlmostEqual(math.log(2), loss)


class TestTrain(unittest.TestCase):
    def test_yields_every_epoch_with_its_average_loss(self):
        network = SkipGramNetwork(["a", "b"], 2, Mulberry32(3))
        trainer = SkipGramTrainer(network, [(0, 1), (1, 0)], [0.5, 1.0], Mulberry32(4), 1)
        results = list(trainer.train(3))
        self.assertEqual([0, 1, 2, 3], [epoch for epoch, _ in results])
        for _, loss in results:
            self.assertGreater(loss, 0.0)

    def test_learning_rate_falls_from_start_to_end(self):
        rates = []
        class RecordingTrainer(SkipGramTrainer):
            def train_epoch(self, lr: float) -> float:
                rates.append(lr)
                return 0.0
        list(RecordingTrainer(SkipGramNetwork(["a"], 1), [], [1.0], Mulberry32(1), 1, 0.4, 0.0).train(4))
        for expected, actual in zip([0.4, 0.3, 0.2, 0.1, 0.0], rates):
            self.assertAlmostEqual(expected, actual)


class TestMulberry32(unittest.TestCase):
    def test_matches_typescript_for_seed_42(self):
        rand = Mulberry32(42)
        self.assertEqual([0.601104, 0.448291, 0.852466, 0.669734, 0.174814], [round(rand(), 6) for _ in range(5)])

    def test_same_seed_gives_same_sequence(self):
        first, second = Mulberry32(7), Mulberry32(7)
        self.assertEqual([first() for _ in range(5)], [second() for _ in range(5)])


if __name__ == "__main__":
    unittest.main()
