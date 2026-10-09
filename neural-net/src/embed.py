from __future__ import annotations
from typing import Callable, Iterator
import math
from dataclasses import dataclass

from bpe import BPETokenizer, Token, TokenFreqSet, TokenList, Vocabulary

## Curated training corpus for Word2Vec Skip-gram and transformer training.
##
## These sentences are designed so that semantically related words frequently
## co-occur within small windows (2-3 words apart). The corpus covers several
## semantic clusters — animals, food, royalty, professions, properties — with
## enough overlap that the model can discover relationships purely from
## co-occurrence statistics.
##
## The royalty sentences use parallel structure ("the king is a man", "the queen
## is a woman") so the model can learn the analogy: king - man + woman ≈ queen.
##
## Tokenization uses BPE (Byte Pair Encoding) trained on this corpus, connecting
## the Skip-gram demo to the BPE tokenizer demo — the same algorithm that splits
## text into subword tokens also defines the vocabulary for embedding training.
 
 # Training data
CORPUS = [
  ## Animals — pets
  "the cat sat on the mat",
  "the dog sat on the rug",
  "a cat is a small pet",
  "a dog is a loyal pet",
  "the cat chased the mouse",
  "the dog chased the cat",
  "the kitten is a baby cat",
  "the puppy is a baby dog",
  "the cat and the dog are pets",
  "a kitten is small and cute",
  "a puppy is small and playful",
  "the cat sleeps on the bed",
  "the dog sleeps on the floor",
  "she loves her pet cat",
  "he loves his pet dog",
  "the cat drinks milk",
  "the dog eats meat",
  "cats and dogs are popular pets",
  "the happy cat purred loudly",
  "the happy dog wagged its tail",

  ## Animals — wild
  "the lion is a wild animal",
  "the tiger is a wild animal",
  "the elephant is a big animal",
  "the mouse is a tiny animal",
  "lions and tigers are big cats",
  "the bear lives in the forest",
  "the wolf lives in the forest",
  "the eagle flies in the sky",
  "the fish swims in the water",
  "birds fly and fish swim",

  ## Food
  "i ate pizza for dinner",
  "she ate pasta for lunch",
  "he ate sushi for dinner",
  "pizza and pasta are italian food",
  "sushi is japanese food",
  "bread and cheese make a sandwich",
  "fruit and vegetables are healthy food",
  "cake and cookies are sweet food",
  "rice is a common food",
  "i love pizza and pasta",
  "she cooked dinner for the family",
  "he made lunch at home",
  "the chef cooked a delicious meal",
  "coffee and tea are hot drinks",
  "juice and water are cold drinks",

  ## Royalty — parallel structure for analogies
  "the king is a man who rules",
  "the queen is a woman who rules",
  "the king sits on the throne",
  "the queen sits on the throne",
  "the prince is the son of the king",
  "the princess is the daughter of the queen",
  "the king and queen rule the kingdom",
  "a prince is a young man of royal blood",
  "a princess is a young woman of royal blood",
  "the king wore a golden crown",
  "the queen wore a silver crown",
  "the prince will become king",
  "the princess will become queen",
  "the man became king of the land",
  "the woman became queen of the land",
  "the king is a powerful man",
  "the queen is a powerful woman",
  "the man was crowned king",
  "the woman was crowned queen",
  "the prince is a boy of noble birth",
  "the princess is a girl of noble birth",
  "the prince is a royal boy",
  "the princess is a royal girl",
  "the young prince played in the castle",
  "the young princess played in the castle",

  ## Professions
  "the doctor works at the hospital",
  "the nurse works at the hospital",
  "the teacher works at the school",
  "the student learns at the school",
  "the chef works in the kitchen",
  "the doctor heals the sick",
  "the teacher helps the student learn",
  "the nurse helps the doctor",
  "the scientist works in the lab",
  "the engineer builds machines",

  ## Properties — size
  "the elephant is very big",
  "the mouse is very small",
  "the cat is small and quick",
  "the dog is big and strong",
  "the lion is big and fierce",
  "the kitten is tiny and cute",
  "the bear is large and powerful",

  ## Actions and relationships
  "the boy and the girl play together",
  "the man and the woman walk together",
  "a boy is a young man",
  "a girl is a young woman",
  "the boy will grow into a man",
  "the girl will grow into a woman",
  "the boy runs fast",
  "the girl runs fast",
  "the man is tall and strong",
  "the woman is tall and smart",
  "he is a kind man",
  "she is a kind woman",
  "the boy helped his father",
  "the girl helped her mother",

  ## Nature
  "the sun shines in the sky",
  "the moon glows at night",
  "stars shine in the dark sky",
  "the river flows to the sea",
  "rain falls from the sky",
  "the tree grows in the forest",
];

STORIES = [
  ## Royal fairy tales
  "once upon a time there was a king who ruled a great kingdom. the king was a tall and powerful man. he sat on a golden throne in a big castle. one day the king said i need a queen to rule with me. so he found a wise and kind woman and she became queen. the queen wore a silver crown and sat on the throne. they ruled the kingdom together and lived happily ever after.",
  "a young prince lived in a castle with the king and queen. the prince was a brave boy who loved to play in the garden. one day he wandered into the deep forest and found a lost princess. the princess was a clever girl of noble birth. the prince said come with me back to the castle. so the prince and the princess walked through the forest together. they returned to the kingdom and the king and queen were happy.",
  "there once was a wicked old man who took the golden crown from the king. the kingdom fell into darkness and the queen was very sad. the young prince said i will find the crown and bring it back. he went on a long journey through the forest and across the river. the brave prince found the old man in a small house by the sea. he took back the crown and returned to the castle. the king wore the crown again and the kingdom was happy once more.",
  "the princess wanted to become queen one day. she was a smart and powerful young woman. the king told her you must be kind and brave to rule. so the princess went to the village and helped the poor and the sick. she gave food to the hungry and water to the old. the people loved her and said she will be a great queen. the princess returned to the castle and the king was proud.",
  "long ago a prince and a princess from a far away kingdom came to the castle. the king and queen said you are welcome here. the prince was a strong young man and the princess was a tall young woman. they sat on the throne and ate a great meal with the royal family. the prince said this is a beautiful kingdom. the king said you may stay as long as you wish. and so they lived happily in the castle.",

  ## Animal fairy tales
  "once upon a time a small cat lived in a village with an old woman. the cat was a loyal pet who sat on the mat by the door. one day the cat chased a tiny mouse into the deep forest. in the forest the cat met a big wild dog. the dog said i am lost and hungry. the kind cat said come with me back to the village. so the cat and the dog walked home together and the old woman gave them both milk and meat.",
  "there was a little kitten who wandered away from home. the kitten was small and cute but very brave. she went through the garden and into the dark forest. there she saw a great bear and a fierce wolf. the kitten was not afraid and said i am looking for my mother. the bear said your mother the cat lives by the river. so the brave little kitten found her mother and they went back home. the puppy next door wagged its tail when they returned.",
  "a loyal dog lived with a boy in a house near the forest. the dog and the boy played together every day. one day they went to the river and saw a fish swim in the water. the boy said i wish i could swim like a fish. the dog jumped into the river and the boy laughed. then an eagle flew across the sky above them. the boy and his dog walked home through the forest as the sun went down.",
  "in a kingdom by the sea there lived a cat and a dog. the cat was quick and small and the dog was big and strong. one day the king said i need a clever pet to help me find my lost crown. the cat and the dog went into the forest together. they found the golden crown under a tall tree. the king was so happy he gave them a great meal of meat and milk. the cat purred loudly and the dog wagged its tail.",
  "once there were three baby animals in the forest. a kitten a puppy and a tiny mouse. they played together near the river every day. the kitten chased the mouse and the puppy chased the kitten. one day a big lion came to the river. the three little animals were afraid but the brave puppy said we are not scared. the lion laughed and said you are small but very brave. the lion walked away and the three friends played happily.",
  "the old woman had a cat who loved to sleep on the bed. one night the cat heard a sound at the door. she went outside and saw a lost baby bird in the garden. the cat was kind and did not chase the bird. she said come inside it is cold and dark. the bird slept on the mat and in the morning the cat took the bird to the tall tree by the river. the bird flew into the sky and the cat went home.",

  ## Village and profession tales
  "once upon a time there was a doctor who worked at the hospital in a small village. one day a sick boy came to the door. the doctor said i will help you. the nurse helped the doctor and they healed the boy. the boy said thank you and went home to his family. his mother made him a warm meal of bread and cheese. the boy ate his dinner and slept on his bed.",
  "a clever young woman became a teacher at the school in the village. she helped every student learn to read and write. one day a boy said i want to become a doctor. the teacher said you must study hard and be brave. the boy worked every day at the school and the teacher was proud. he grew into a tall strong man and went to work at the hospital. the teacher said i am happy i could help.",
  "the chef worked in the kitchen of the castle. he cooked a delicious meal for the king and queen every day. one day the king said i want pizza and pasta for dinner. the chef said but pizza is italian food and pasta is italian food too. the king laughed and said i love italian food. so the chef made the best pizza and pasta in the kingdom. the queen ate sushi and rice because she loved japanese food.",
  "there was a scientist who worked in a lab near the forest. the scientist was a clever old man who loved animals. one day he found a sick baby bear by the river. he took the bear to the doctor at the hospital. the nurse helped the doctor heal the baby bear. when the bear was strong again the scientist took him back to the forest. the bear was happy and the scientist smiled.",
  "a young man wanted to become an engineer. he went to the school in the village and studied hard. the teacher said you are a smart student. one day the engineer built a great machine. the king heard about the machine and said come to the castle. the engineer went to the castle and showed the king his machine. the king said you are a clever man and gave him a golden crown. the engineer lived happily in the kingdom.",

  ## Nature and journey tales
  "once upon a time the sun and the moon had a race across the sky. the sun shines bright in the day and the moon glows at night. the sun said i am faster than you. the moon said but the stars shine with me. they raced from the river to the sea. the sun ran fast but the moon was clever and took a path through the dark sky. in the end they said we are both great and they lived happily together.",
  "a brave girl wandered into the deep forest one day. the tall trees grew thick and the sky turned dark. rain fell from the sky and the river flowed fast. the girl found a small house by the river. an old woman lived there with a loyal cat and a playful dog. the old woman said come inside and have some food. she gave the girl bread and milk and the girl slept by the fire. in the morning the girl returned home to her family.",
  "there once was a great tree that grew in the middle of the forest. birds lived in the tree and a bear slept under it. the river flowed nearby and fish swam in the water. one day a young boy found the tree and said this is a beautiful place. he came back every day and the animals were not afraid. the eagle flew down from the sky and sat on the boy his arm. the boy and the animals became friends and lived happily in the forest.",
  "long ago rain did not fall from the sky and the river was dry. the animals in the forest were very hungry and sad. the lion said we must find water. the wolf said i know a path to the sea. so the lion the wolf and the bear went on a long journey. they walked through the forest and over the land. at last they found a great river that flowed to the sea. the animals drank the cold water and were happy.",
  "the moon glowed bright one night and the stars shined in the dark sky. a little girl sat by the river and looked up. she said i wish i could fly like an eagle. then a great bird came down from the sky and said climb on my back. the girl flew over the forest and the river and the sea. she saw the king castle and the small village below. the bird took her home and she told her mother about the journey. her mother said that is a beautiful story.",

  ## Food and feast tales
  "the king and queen had a great feast at the castle. the chef cooked pizza and pasta and sushi and rice. there was bread and cheese and cake and cookies on the table. the prince ate meat and the princess drank cold juice. the boy and the girl from the village came to the feast. they ate fruit and vegetables and sweet food. coffee and tea and cold water were the drinks. the king said this is the best meal in the kingdom.",
  "once there was a poor old woman who had no food. she was very hungry and sad. one day a kind man came to her door with bread and cheese. he said i am a chef and i made this food for you. the old woman ate the bread and cheese and was happy. the next day the chef came back with pasta and rice and cake. the old woman said you are the most kind man i have ever met. they became friends and ate dinner together every day.",
  "a boy and a girl went to the forest to find food for their family. they found fruit on the trees and vegetables in a garden. the girl said we should bring some back for mother and father. they walked home through the forest with the food. their mother cooked a delicious meal of rice and meat. the family sat together and ate dinner. the father said you are brave and clever children. they lived happily in their small house by the river.",

  ## Mixed theme tales
  "once upon a time a young prince had a pet cat and a pet dog. the cat was small and quick and the dog was big and loyal. one day the prince took his pets to the forest. the cat chased a mouse and the dog chased the cat. the prince laughed and said you two are very playful. they sat by the river and the prince ate bread and cheese for lunch. the sun shined in the sky and the prince and his pets were happy.",
  "there was a princess who loved animals. she had a kitten and a puppy in the castle. one day she found a baby bird in the garden. she asked the doctor to help the sick bird. the doctor and the nurse healed the bird. the princess was happy and gave the doctor a golden crown. the kitten and the puppy played with the bird in the castle. the king and queen said our daughter is a kind and brave girl.",
  "long ago a wise old man lived in a house by the sea. he had a loyal dog and a clever cat. every day he sat on a mat by the door and watched the sun shine over the water. one day a young boy came to his door and said i am lost. the old man gave the boy food and water and said you can stay here. the boy lived with the old man and helped him every day. the cat slept on the bed and the dog slept on the floor. they all lived happily together.",
  "a brave young woman left the village and went on a journey to find the lost kingdom. she walked through the deep forest and across the great river. she met a lion who said i will help you. they found the kingdom hidden behind a tall tree. the old castle was dark and a wicked man sat on the throne. the brave woman and the lion chased the wicked man away. she became queen and ruled the kingdom with kindness. the lion lived in the castle garden and they were happy ever after.",
  "once upon a time in a small village there lived a boy and a girl. the boy wanted to become a king and the girl wanted to become a queen. they went to the castle and asked the old king for help. the king said you must be brave and kind and smart. the boy and the girl went on a long journey through the forest. they helped the animals and the poor people they found. when they returned the king said you are ready. the boy became king and the girl became queen and they ruled the kingdom together.",

  ## Additional tales — covering remaining vocabulary
  "the king had a large elephant that lived in the castle garden. the elephant was a big and powerful animal. one day a fierce tiger came from the wild forest. the tiger runs fast and the elephant was afraid. the brave prince said i will help you. the lion and the wolf came to help too. the tiger saw the lions and tigers together and ran away. the elephant was happy and the prince was crowned a hero. the king sits on the throne and rules the kingdom with the prince who will grow into a great man.",
  "every morning the boy runs to the school in the village. his dog runs with him and they are both very fast. the teacher helps the student learn and the nurse helps the doctor at the hospital. the chef works in the kitchen and makes a common meal of rice and bread. hot coffee and tea are popular drinks in the village. the boy eats his lunch of meat and a sandwich and drinks cold juice. then he walks home and his cat sleeps on the rug by the door. his dogs sit on the floor and make a loud sound.",
  "the old man and the old woman loved to walk by the river. the river flows to the sea and the tree grows on the land beside it. birds fly in the sky and the eagle flies above them all. the fish swims in the water and the bear lives in the forest nearby. the old woman loves her pet cats and dogs. they are popular pets in the village. the old man builds machines and works as an engineer. their son learns at the school and their daughter heals the sick at the hospital.",
  "a baby elephant and a baby tiger played in the forest. the elephant was big and the tiger was fierce but they were friends. the elephant eats fruit and vegetables which are healthy food. the tiger eats meat because it is a wild animal. one day rain falls from the sky and the river flows very fast. the two animals sat under a large tree and the bear sat with them. the wolf sat nearby and the eagle sat in the tall tree above. when the sun came back they all played happily together.",
  "the princess had a kitten that sleeps on her bed and a puppy that sleeps on the floor. the kitten is tiny and cute and the puppy is small and playful. every day the princess walks with her pets in the garden. the kitten runs and the puppy runs after her. they are both popular pets in the kingdom. the queen loves her daughter and said she grows into a brave young woman. the king said our daughter is of royal blood and noble birth. she is a true animal lover. one day she will rule this land.",
];

# /** Build a vocabulary from the corpus — returns word↔index mappings sorted by frequency (most common first). */
# export function buildVocab(corpus: string[]) {
#   const freq = new Map<string, number>();
#   for (const sentence of corpus) {
#     for (const word of tokenize(sentence)) {
#       freq.set(word, (freq.get(word) ?? 0) + 1);
#     }
#   }

#   const indexToWord: string[] = [];
#   const wordToIndex = new Map<string, number>();

#   for (const [word] of [...freq.entries()].sort((a, b) => b[1] - a[1])) {
#     wordToIndex.set(word, indexToWord.length);
#     indexToWord.push(word);
#   }

#   return { wordToIndex, indexToWord, freq };
# }

# Options for training the skip-gram model.
@dataclass(frozen=True)
class TrainingOptions:
    words: list[str] # The list of words in the corpus in the order they appear
    epochs: int # Number of times to iterate over the entire corpus during training
    vector_size: int # Length of the embedding vector for each word
    window_size: int # How many words to consider to the left and right of the target word
    negative_samples: int # Number of negative samples to draw for each positive (target, context) pair

@dataclass(frozen=True)
class InitResult:
    vocabSize: int
    sentenceCount: int
    embeddingDim: int
    windowSize:int
    totalPairs:int
    
@dataclass(frozen=True)
class EpochResult:
    epoch: int
    loss: float
    
@dataclass(frozen=True)
class WordEmbedding:
    word: str
    vector: list[float]
    
@dataclass(frozen=True)
class NeighborScore:
    word: str
    score: float

@dataclass(frozen=True)
class Neighbor:
    word: str
    nearest: list[NeighborScore]
    
@dataclass(frozen=True)
class SimilarityPair:
    a: str
    b: str
    score: float

@dataclass(frozen=True)
class Analogy:
    query: str
    result: str
    score: float
    
@dataclass(frozen=True)
class TrainEmbedResult:
    embeddings: list[WordEmbedding]
    neightbors: list[Neighbor]
    similarities: list[SimilarityPair]
    analogies: list[Analogy]
    warnings: list[str]
    

# One token's vector of weights: a fixed number of values (the vector size) that are
# adjusted during training. Tokens used in similar contexts end up with vectors that
# point in similar directions.
#
# Unlike the token classes, this is not frozen: training changes the values in place.
#
# Example, vector size 3:
#
#   a = EmbeddingVector([1.0, 2.0, 0.5]),  b = EmbeddingVector([0.5, -1.0, 2.0])
#   a.dot(b)             ->  1.0*0.5 + 2.0*-1.0 + 0.5*2.0 = -0.5
#   a.add_scaled(b, 0.1) ->  a becomes [1.05, 1.9, 0.7]
class EmbeddingVector:
    def __init__(self, values: list[float]):
        self.values = values

    def size(self) -> int:
        return len(self.values)

    # How strongly two vectors agree: large and positive when they point the same way,
    # near zero when unrelated, negative when they point in opposite directions.
    # (Adds the products one at a time, in order, rather than using sum(), whose
    # extra-precise float addition in newer Pythons would give slightly different results
    # from the TypeScript.)
    def dot(self, other: EmbeddingVector) -> float:
        result = 0.0
        for d in range(len(self.values)):
            result += self.values[d] * other.values[d]
        return result

    # Moves this vector by factor * other: towards other if factor is positive,
    # away from it if negative. other is not changed.
    def add_scaled(self, other: EmbeddingVector, factor: float) -> None:
        for d in range(len(self.values)):
            self.values[d] += factor * other.values[d]

    # A separate vector with the same values, so changing one does not change the other
    def copy(self) -> EmbeddingVector:
        return EmbeddingVector(list(self.values))

    def __repr__(self) -> str:
        return f"EmbeddingVector({self.values})"

# Trains word embeddings on CORPUS: learns BPE tokens, builds a fixed vocabulary, then
# trains a skip-gram model with negative sampling (see train_skip_gram).
class Embed:
    def __init__(self):
        self.tokenizer = BPETokenizer()
        self.token_freq_set: TokenFreqSet = self.tokenizer.train_bpe(" ".join(CORPUS).lower(), 500)
        self.vocabulary = Vocabulary(self.token_freq_set)

    def tokenize(self, text: str) -> TokenList:
        return self.tokenizer.tokenize(text.lower())

    # Seendable random number generator
    def mulberry32(self, seed: int) -> Callable[[], float]:
        MASK = 0xFFFFFFFF  # keep values to 32 bits, like JavaScript
        state = int(seed) & MASK

        def next_random() -> float:
            nonlocal state
            state = (state + 0x6D2B79F5) & MASK
            t = ((state ^ (state >> 15)) * (1 | state)) & MASK
            t = ((t + (((t ^ (t >> 7)) * (61 | t)) & MASK)) & MASK) ^ t
            return ((t ^ (t >> 14)) & MASK) / 4294967296

        return next_random

    # An "activation function" - decides how strongly a neuron fires.
    # squashes input into the range 0 to 1
    def sigmoid(self, x: float) -> float:
        if x > 6:
            return 1.0
        if x < -6:
            return 0.0
        return 1 / (1 + math.exp(-x))

    # Builds the skip-gram training pairs: (target, context) vocabulary indices.
    # Every token is paired with every other token within window_size positions of it in
    # the same sentence. Pairs never cross from one sentence into the next.
    #
    # sentences: each sentence already encoded as vocabulary indices
    #
    # Example, with one sentence [9, 38, 249] ("the cat sat") and window_size = 1:
    #
    #   i=0 (9):    (9, 38)
    #   i=1 (38):   (38, 9), (38, 249)
    #   i=2 (249):  (249, 38)
    def build_training_pairs(self, sentences: list[list[int]], window_size: int) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for indices in sentences:
            for i in range(len(indices)):
                for j in range(max(0, i - window_size), min(len(indices) - 1, i + window_size) + 1):
                    if j != i:
                        pairs.append((indices[i], indices[j]))
        return pairs

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
    def build_negative_sampling_table(self, vocabulary: Vocabulary) -> list[float]:
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
    def sample_negative(self, cumulative: list[float], rand: Callable[[], float]) -> int:
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

    # Initialize the two weight matrices with small random values, each between
    # -scale/2 and +scale/2, where scale = 0.5 / dim. Starting small and random keeps every
    # token's vector different but close to zero, so no token starts out favoured.
    #
    # w_in holds each token's embedding as a target (the vectors we keep at the end);
    # w_out holds each token's vector as a context. Each is a list with one
    # EmbeddingVector of dim values per vocabulary index, so w_in[9] is the target
    # vector for the token at index 9.
    #
    # Example, with vocab_size = 3 and dim = 2:
    #
    #   w_in = [EmbeddingVector([t0d0, t0d1]),    <- token 0
    #           EmbeddingVector([t1d0, t1d1]),    <- token 1
    #           EmbeddingVector([t2d0, t2d1])]    <- token 2
    #
    # The rand() calls alternate between w_in and w_out, value by value, as in the
    # TypeScript, so the same seed gives the same starting weights in both versions.
    def init_weights(self, vocab_size: int, dim: int, rand: Callable[[], float]) -> tuple[list[EmbeddingVector], list[EmbeddingVector]]:
        scale = 0.5 / dim
        w_in: list[EmbeddingVector] = []
        w_out: list[EmbeddingVector] = []
        for _ in range(vocab_size):
            in_values: list[float] = []
            out_values: list[float] = []
            for _ in range(dim):
                in_values.append((rand() - 0.5) * scale)
                out_values.append((rand() - 0.5) * scale)
            w_in.append(EmbeddingVector(in_values))
            w_out.append(EmbeddingVector(out_values))
        return w_in, w_out

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
    def shuffle(self, arr: list[tuple[int, int]], rand: Callable[[], float]) -> None:
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
    def learning_rate(self, epoch: int, epochs: int, lr_start: float, lr_end: float) -> float:
        if epochs == 0:
            return lr_start
        return lr_start - (lr_start - lr_end) * (epoch / epochs)

    # Trains on one (target, other) pair and returns its loss.
    #
    # label is 1 for a positive pair (other really is near target in the text) and 0 for a
    # negative sample (other is a random token). The model's score for the pair is
    # sigmoid(target's w_in vector . other's w_out vector), a value from 0 to 1.
    # Training moves the score towards the label:
    #
    #   positive (label 1):  grad = lr * (1 - score)  ->  the two vectors move closer together
    #   negative (label 0):  grad = lr * (0 - score)  ->  the two vectors move apart
    #
    # Both vectors are updated using each other's values from before the update, which is
    # why a copy of the target vector is taken first.
    #
    # The loss measures how wrong the score was: -log(score) for a positive and
    # -log(1 - score) for a negative. It is 0 for a perfect score and grows as the score
    # gets worse. 1e-10 stops log(0) when a score is exactly 0 or 1.
    #
    # Example, positive pair, vector size 2, lr = 0.1:
    #
    #   target vector (w_in) = [0.5, 0.5],  other vector (w_out) = [0.5, -0.5]
    #   dot = 0.25 - 0.25 = 0  ->  score = sigmoid(0) = 0.5  ->  grad = 0.1 * (1 - 0.5) = 0.05
    #   target vector becomes [0.5 + 0.05*0.5, 0.5 + 0.05*-0.5] = [0.525, 0.475]
    #   other vector becomes  [0.5 + 0.05*0.5, -0.5 + 0.05*0.5] = [0.525, -0.475]
    #   loss = -log(0.5) = 0.693
    def train_pair(self, w_in: list[EmbeddingVector], w_out: list[EmbeddingVector],
                   target: int, other: int, label: int, lr: float) -> float:
        target_vector = w_in[target]
        other_vector = w_out[other]

        score = self.sigmoid(target_vector.dot(other_vector))
        grad = lr * (label - score)

        old_target_vector = target_vector.copy()
        target_vector.add_scaled(other_vector, grad)
        other_vector.add_scaled(old_target_vector, grad)

        if label == 1:
            return -math.log(score + 1e-10)
        return -math.log(1 - score + 1e-10)

    # Runs one epoch: shuffles the pairs, then for each (target, context) pair trains on the
    # positive pair and on negative_samples random negatives. Returns the average loss per pair.
    #
    # A negative that happens to be the real context is skipped, since it isn't really "wrong".
    def train_epoch(self, pairs: list[tuple[int, int]], w_in: list[EmbeddingVector], w_out: list[EmbeddingVector],
                    negative_samples: int, cumulative: list[float], rand: Callable[[], float],
                    lr: float) -> float:
        if not pairs:
            return 0.0

        total_loss = 0.0
        self.shuffle(pairs, rand)

        for target, context in pairs:
            # Positive sample: push target and context closer
            total_loss += self.train_pair(w_in, w_out, target, context, 1, lr)

            # Negative samples: push target and random tokens apart
            for _ in range(negative_samples):
                neg = self.sample_negative(cumulative, rand)
                if neg == context:
                    continue
                total_loss += self.train_pair(w_in, w_out, target, neg, 0, lr)

        return total_loss / len(pairs)

    # Rounds to 6 decimal places, rounding halves up like JavaScript's Math.round.
    # (Python's round() rounds halves to the nearest even digit instead.)
    def round_6(self, value: float) -> float:
        return math.floor(value * 1000000 + 0.5) / 1000000

    # Rounds to 2 decimal places, rounding halves up like JavaScript's Math.round (see round_6)
    def round_2(self, value: float) -> float:
        return math.floor(value * 100 + 0.5) / 100

    # How closely two vectors point the same way, ignoring their lengths:
    # 1 for the same direction, 0 for unrelated, -1 for opposite.
    # Returns 0 if either vector is all zeros, since it then has no direction.
    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        vector_a = EmbeddingVector(a)
        vector_b = EmbeddingVector(b)
        length_product = math.sqrt(vector_a.dot(vector_a)) * math.sqrt(vector_b.dot(vector_b))
        if length_product == 0:
            return 0.0
        return vector_a.dot(vector_b) / length_product

    # A word's embedding as a plain list, rounded to 6 decimal places
    def get_vector(self, w_in: list[EmbeddingVector], word_index: int) -> list[float]:
        return [self.round_6(value) for value in w_in[word_index].values]

    # Looks up each query word in the vocabulary. Returns the indexes of words that are
    # single BPE tokens, plus a warning for each word that splits into several tokens.
    def find_query_indices(self, words: list[str]) -> tuple[list[int], list[str]]:
        warnings: list[str] = []
        query_indices: list[int] = []

        for word in words:
            index = self.vocabulary.token_to_index.get(Token(word.lower()))
            if index is None:
                bpe_tokens = [token.token for token in self.tokenizer.split_word(word.lower()).entries]
                warnings.append(f'"{word}" is not a single BPE token — it splits into [{", ".join(bpe_tokens)}]')
            else:
                query_indices.append(index)

        return query_indices, warnings

    # Trains the skip-gram model, yielding results as it goes: an InitResult once the
    # training pairs are built, then an EpochResult every `step` epochs (and for the last epoch).
    def train_skip_gram(self, opts: TrainingOptions) -> Iterator[InitResult | EpochResult | TrainEmbedResult]:
        words = opts.words
        epochs = opts.epochs
        embeded_vector_size = opts.vector_size
        window_size = opts.window_size
        negative_samples = opts.negative_samples

        # Uses the tokenizer and vocabulary built in __init__
        vocab_size = self.vocabulary.size()
        rand = self.mulberry32(42)

        # Encode each sentence as vocabulary indices, then pair up nearby tokens
        sentences_as_indices = [self.vocabulary.encode(self.tokenize(sentence)) for sentence in CORPUS]
        positive_pairs = self.build_training_pairs(sentences_as_indices, window_size)
        print(f"Total training pairs: {len(positive_pairs)}")

        yield  InitResult(
            vocabSize=vocab_size,
            sentenceCount=len(CORPUS),
            embeddingDim=embeded_vector_size,
            windowSize=window_size,
            totalPairs=len(positive_pairs)
        )

        negative_sampling_table = self.build_negative_sampling_table(self.vocabulary)
        w_in, w_out = self.init_weights(vocab_size, embeded_vector_size, rand)

        # How often (in epochs) to report progress: about 50 reports over the whole run,
        # but never less than every epoch. For example, 200 epochs -> every 4th epoch;
        # 30 epochs -> every epoch.
        step = max(1, epochs // 50)

        # The learning rate starts at lr_start and falls to lr_end over training, so early
        # updates make big changes and later ones make fine adjustments.
        lr_start = 0.025
        lr_end = 0.001

        # Note: epochs + 1 passes (0 to epochs inclusive), as in the TypeScript
        for epoch in range(epochs + 1):
            lr = self.learning_rate(epoch, epochs, lr_start, lr_end)
            loss = self.train_epoch(positive_pairs, w_in, w_out, negative_samples, negative_sampling_table, rand, lr)
            if epoch % step == 0 or epoch == epochs:
                yield EpochResult(epoch=epoch, loss=self.round_6(loss))

        # Report on the query words that are single tokens; the rest only get a warning
        query_indices, warnings = self.find_query_indices(words)

        embeddings = [WordEmbedding(word=self.vocabulary.token_at(index).token, vector=self.get_vector(w_in, index))
                      for index in query_indices]

        # Nearest neighbors for each query word: the 5 other tokens with the most similar vectors
        neighbors: list[Neighbor] = []
        for query_index in query_indices:
            query_vector = self.get_vector(w_in, query_index)
            scores: list[NeighborScore] = []
            for index in range(vocab_size):
                if index == query_index:
                    continue
                scores.append(NeighborScore(
                    word=self.vocabulary.token_at(index).token,
                    score=self.round_2(self.cosine_similarity(query_vector, self.get_vector(w_in, index)))
                ))
            # sort() is stable, like JavaScript's, so tied scores keep vocabulary order
            scores.sort(key=lambda neighbor_score: neighbor_score.score, reverse=True)
            neighbors.append(Neighbor(word=self.vocabulary.token_at(query_index).token, nearest=scores[:5]))

        # Pairwise similarity between query words
        similarities: list[SimilarityPair] = []
        for i in range(len(query_indices)):
            for j in range(i + 1, len(query_indices)):
                similarities.append(SimilarityPair(
                    a=self.vocabulary.token_at(query_indices[i]).token,
                    b=self.vocabulary.token_at(query_indices[j]).token,
                    score=self.round_2(self.cosine_similarity(self.get_vector(w_in, query_indices[i]),
                                                              self.get_vector(w_in, query_indices[j])))
                ))

        # TODO: analogies, once the rest of the TypeScript is ported
        yield TrainEmbedResult(
            embeddings=embeddings,
            neightbors=neighbors,
            similarities=similarities,
            analogies=[],
            warnings=warnings
        )
