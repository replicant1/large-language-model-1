import unittest
import io
import contextlib
from typing import Optional
from embed import Embed
from embed import TrainingOptions
from embed import InitResult
from embed import EpochResult
from embed import TrainEmbedResult
from skip_gram_network import SkipGramNetwork

# Creating an Embed trains the BPE tokenizer on the corpus, which is slow and prints a lot,
# so the tests that need one share a single Embed, created the first time and its output hidden.
class EmbedTestCase(unittest.TestCase):
    shared_embed: Optional[Embed] = None

    @classmethod
    def setUpClass(cls):
        if EmbedTestCase.shared_embed is None:
            with contextlib.redirect_stdout(io.StringIO()):
                EmbedTestCase.shared_embed = Embed()
        cls.embed = EmbedTestCase.shared_embed


class TestDot(unittest.TestCase):
    def test_example_from_comment(self):
        self.assertAlmostEqual(-0.5, Embed.dot([1.0, 2.0, 0.5], [0.5, -1.0, 2.0]))

    def test_is_symmetric(self):
        a = [0.3, -0.7, 1.1]
        b = [2.0, 0.4, -0.2]
        self.assertEqual(Embed.dot(a, b), Embed.dot(b, a))


class TestBuildTrainingPairs(EmbedTestCase):
    def test_window_of_one(self):
        pairs = self.embed.build_training_pairs([[9, 38, 249]], 1)
        self.assertEqual([(9, 38), (38, 9), (38, 249), (249, 38)], pairs)

    def test_window_of_two(self):
        pairs = self.embed.build_training_pairs([[1, 2, 3, 4]], 2)
        self.assertEqual([
            (1, 2), (1, 3),
            (2, 1), (2, 3), (2, 4),
            (3, 1), (3, 2), (3, 4),
            (4, 2), (4, 3),
        ], pairs)

    def test_window_larger_than_sentence(self):
        pairs = self.embed.build_training_pairs([[1, 2]], 5)
        self.assertEqual([(1, 2), (2, 1)], pairs)

    def test_pairs_do_not_cross_sentences(self):
        pairs = self.embed.build_training_pairs([[1, 2], [3, 4]], 2)
        self.assertEqual([(1, 2), (2, 1), (3, 4), (4, 3)], pairs)

    def test_repeated_tokens_are_kept(self):
        pairs = self.embed.build_training_pairs([[5, 5]], 1)
        self.assertEqual([(5, 5), (5, 5)], pairs)

    def test_empty_and_single_token_sentences(self):
        self.assertEqual([], self.embed.build_training_pairs([[], [7]], 2))


class TestSimilarityToEveryToken(EmbedTestCase):
    def test_scores_every_other_token_in_vocabulary_order(self):
        network = SkipGramNetwork(["a", "b", "c"], 2)
        network.set_embedding(0, [1.0, 0.0])
        network.set_embedding(1, [0.0, 1.0])
        network.set_embedding(2, [-1.0, 0.0])
        self.assertEqual([(0, 1.0), (2, -1.0)], self.embed.similarity_to_every_token(network, [1.0, 0.0], {1}))

    def test_token_embeddings_are_rounded_first(self):
        # 0.1234564 rounds to 0.123456, so the similarity uses that, not the raw weight
        network = SkipGramNetwork(["a"], 1)
        network.set_embedding(0, [0.1234564])
        [(index, similarity)] = self.embed.similarity_to_every_token(network, [2.0], set())
        self.assertEqual(0, index)
        self.assertEqual(self.embed.cosine_similarity([2.0], [0.123456]), similarity)


class TestRoundLikeJs(EmbedTestCase):
    def test_rounds_to_six_places(self):
        self.assertEqual(2.130895, self.embed.round_like_js(2.1308954, 6))
        self.assertEqual(2.130896, self.embed.round_like_js(2.1308956, 6))

    def test_rounds_to_two_places(self):
        self.assertEqual(0.96, self.embed.round_like_js(0.9642, 2))
        self.assertEqual(-0.24, self.embed.round_like_js(-0.2422, 2))

    def test_halves_round_up_like_javascript(self):
        # Python's round(0.5) is 0 (round half to even); Math.round(0.5) is 1
        self.assertEqual(0.000001, self.embed.round_like_js(0.0000005, 6))
        self.assertEqual(0.13, self.embed.round_like_js(0.125, 2))


class TestTrainSkipGram(unittest.TestCase):
    def test_yields_init_then_epoch_results(self):
        with contextlib.redirect_stdout(io.StringIO()):
            results = list(Embed().train_skip_gram(TrainingOptions([], 3, 4, 2, 2)))
        self.assertIsInstance(results[0], InitResult)
        self.assertEqual(4, results[0].embeddingDim)
        epoch_results = results[1:-1]
        self.assertEqual([0, 1, 2, 3], [result.epoch for result in epoch_results])
        for result in epoch_results:
            self.assertIsInstance(result, EpochResult)
        self.assertLess(epoch_results[-1].loss, epoch_results[0].loss)
        self.assertIsInstance(results[-1], TrainEmbedResult)

    def test_matches_typescript_losses(self):
        # Reference losses from running the TypeScript training loop on the same pairs
        # and sampling table, 10 epochs, dim 8, window 2, 3 negative samples
        with contextlib.redirect_stdout(io.StringIO()):
            results = list(Embed().train_skip_gram(TrainingOptions([], 10, 8, 2, 3)))
        self.assertEqual(
            [2.75102, 2.748616, 2.709061, 2.584951, 2.433389, 2.324782, 2.24623, 2.192255, 2.158386, 2.149308, 2.130895],
            [result.loss for result in results[1:-1]])


if __name__ == "__main__":
    unittest.main()
