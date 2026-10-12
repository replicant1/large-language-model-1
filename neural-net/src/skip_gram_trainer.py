import math
from typing import Callable, Iterator
from bpe import Vocabulary
from skip_gram_network import SkipGramNetwork


class SkipGramTrainer:
    """Trains a SkipGramNetwork on (target, context) pairs, using negative sampling.

    network: the network to train; its weights are updated in place
    pairs: the real (target, context) pairs; shuffled in place at the start of every epoch
    negative_sampling_table: the cumulative table from build_negative_sampling_table
    rand: the random number generator, used for shuffling and for picking negatives
    negative_samples: how many random negatives to train on for each real pair
    lr_start, lr_end: the learning rate falls in a straight line from lr_start to lr_end
        over training, so early updates make big changes and later ones make fine adjustments
    """

    def __init__(self, network: SkipGramNetwork, pairs: list[tuple[int, int]],
                 negative_sampling_table: list[float], rand: Callable[[], float],
                 negative_samples: int, lr_start: float = 0.025, lr_end: float = 0.001):
        self.network = network
        self.pairs = pairs
        self.negative_sampling_table = negative_sampling_table
        self.rand = rand
        self.negative_samples = negative_samples
        self.lr_start = lr_start
        self.lr_end = lr_end

    # Builds the table used to pick negative samples, returned as a running total
    # (cumulative) of each token's probability of being picked.
    #
    # Negative sampling a): for each real (target, context) pair, training also picks a few
    # random tokens as "wrong" contexts, so the model learns which tokens do NOT go together.
    # This builds the probability of each token being picked as one of those negatives.
    #
    # Each token's weight is its frequency raised to the power 0.75 (Mikolov's trick from
    # word2vec), then all weights are divided by their total so they add up to 1.
    # The 0.75 power flattens the differences: common tokens are still picked more often,
    # but rare tokens get a bigger share than their raw count would give them.
    #
    # Example with three tokens:
    #
    #   token   frequency   raw share   frequency ** 0.75   unigram_power (final share)
    #   the     100         90.1%       31.62               0.827  (82.7%)
    #   cat     10           9.0%        5.62               0.147  (14.7%)
    #   mat     1            0.9%        1.00               0.026   (2.6%)
    #
    # Negative sampling b): Running total of unigram_power, used to pick a negative token at random.
    # Each token "owns" a stretch of the range 0 to 1 as wide as its probability, so drawing
    # a random number r in that range and finding the first entry in cumulative that is >= r
    # picks each token with exactly its probability.
    #
    # Example, continuing from above:
    #
    #   token           the     cat     mat
    #   unigram_power   0.827   0.147   0.026
    #   cumulative      0.827   0.974   1.000
    #
    #   r = 0.50  ->  first entry >= 0.50 is 0.827  ->  the
    #   r = 0.90  ->  first entry >= 0.90 is 0.974  ->  cat
    #   r = 0.99  ->  first entry >= 0.99 is 1.000  ->  mat
    #
    # (The TypeScript comment calls this an "alias table for O(1) sampling", but it is a
    # cumulative table: finding the entry takes O(log n) with a binary search.)
    @staticmethod
    def build_negative_sampling_table(vocabulary: Vocabulary) -> list[float]:
        vocab_size = vocabulary.size()

        unigram_power: list[float] = [0.0] * vocab_size
        unigram_sum = 0.0
        for i in range(vocab_size):
            count = vocabulary.frequency_at(i)
            unigram_power[i] = count ** 0.75
            unigram_sum += unigram_power[i]
        for i in range(vocab_size):
            unigram_power[i] /= unigram_sum

        cumulative: list[float] = [0.0] * vocab_size
        cumulative[0] = unigram_power[0]
        for i in range(1, vocab_size):
            cumulative[i] = cumulative[i - 1] + unigram_power[i]
        return cumulative

    # Picks one token index at random to use as a negative, with each token's chance given
    # by the negative sampling table. Draws r in the range 0 to 1, then binary-searches
    # cumulative for the first entry >= r.
    #
    # Example, with cumulative = [0.827, 0.974, 1.000] and r = 0.90:
    #
    #   lo=0, hi=2, mid=1:  cumulative[1] = 0.974 >= 0.90  ->  hi = 1
    #   lo=0, hi=1, mid=0:  cumulative[0] = 0.827 <  0.90  ->  lo = 1
    #   lo == hi == 1       ->  returns 1 (cat)
    #
    # hi starts at the last index, so the result is always a valid index, even if
    # rounding leaves the last cumulative entry slightly below 1.
    @staticmethod
    def sample_negative(cumulative: list[float], rand: Callable[[], float]) -> int:
        r = rand()
        lo = 0
        hi = len(cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # Fisher-Yates shuffle: puts the pairs into a random order, in place.
    # Working back from the last position, swap each item with a randomly chosen
    # item at or before it. Every ordering is equally likely.
    #
    # Example, shuffling [A, B, C, D]:
    #
    #   i=3: j=1  ->  swap D and B  ->  [A, D, C, B]
    #   i=2: j=2  ->  C stays put   ->  [A, D, C, B]
    #   i=1: j=0  ->  swap D and A  ->  [D, A, C, B]
    #
    # Uses rand() rather than Python's random.shuffle, so the order depends only on the seed.
    @staticmethod
    def shuffle(arr: list[tuple[int, int]], rand: Callable[[], float]) -> None:
        for i in range(len(arr) - 1, 0, -1):
            j = math.floor(rand() * (i + 1))
            arr[i], arr[j] = arr[j], arr[i]

    # Learning rate for a given epoch: falls in a straight line from lr_start at epoch 0
    # to lr_end at the last epoch.
    #
    # Example, with lr_start = 0.025, lr_end = 0.001 and epochs = 4:
    #
    #   epoch   0       1       2       3       4
    #   lr      0.025   0.019   0.013   0.007   0.001
    @staticmethod
    def learning_rate(epoch: int, epochs: int, lr_start: float, lr_end: float) -> float:
        if epochs == 0:
            return lr_start
        return lr_start - (lr_start - lr_end) * (epoch / epochs)

    # Trains for the given number of epochs, yielding (epoch, average loss per pair) after each one.
    # Note: epochs + 1 passes (0 to epochs inclusive), as in the TypeScript.
    def train(self, epochs: int) -> Iterator[tuple[int, float]]:
        for epoch in range(epochs + 1):
            lr = SkipGramTrainer.learning_rate(epoch, epochs, self.lr_start, self.lr_end)
            yield epoch, self.train_epoch(lr)

    # Runs one epoch: shuffles the pairs, then for each (target, context) pair trains on the
    # positive pair and on negative_samples random negatives. Returns the average loss per pair.
    #
    # A negative that happens to be the real context is skipped, since it isn't really "wrong".
    def train_epoch(self, lr: float) -> float:
        if not self.pairs:
            return 0.0

        total_loss = 0.0
        SkipGramTrainer.shuffle(self.pairs, self.rand)

        for target, context in self.pairs:
            # Positive sample: push target and context closer
            total_loss += self.network.train_pair(target, context, 1, lr)

            # Negative samples: push target and random tokens apart
            for _ in range(self.negative_samples):
                neg = SkipGramTrainer.sample_negative(self.negative_sampling_table, self.rand)
                if neg == context:
                    continue
                total_loss += self.network.train_pair(target, neg, 0, lr)

        return total_loss / len(self.pairs)
