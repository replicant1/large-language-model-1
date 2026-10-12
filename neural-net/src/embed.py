from __future__ import annotations
from typing import Iterator
import json
import math
from dataclasses import dataclass
from pathlib import Path

from bpe import BPETokenizer, Token, TokenFreqSet, TokenList, Vocabulary
from skip_gram_network import SkipGramNetwork
from skip_gram_trainer import SkipGramTrainer, build_negative_sampling_table
from random_numbers import mulberry32

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

# Analogies to test after training, as (a, b, c) meaning "a - b + c = ?".
# For example, king - man + woman should come out close to queen.
ANALOGIES: list[tuple[str, str, str]] = [
  ("king", "man", "woman"),
  ("queen", "woman", "man"),
  ("prince", "boy", "girl"),
  ("kitten", "cat", "dog"),
  ("puppy", "dog", "cat"),
  ("he", "man", "woman"),
  ("his", "man", "woman"),
]

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
    

# How strongly two vectors agree: large and positive when they point the same way,
# near zero when unrelated, negative when they point in opposite directions.
# (Adds the products one at a time, in order, rather than using sum(), whose
# extra-precise float addition in newer Pythons would give slightly different results
# from the TypeScript.)
#
# Example:  dot([1.0, 2.0, 0.5], [0.5, -1.0, 2.0])  ->  1.0*0.5 + 2.0*-1.0 + 0.5*2.0 = -0.5
def dot(a: list[float], b: list[float]) -> float:
    result = 0.0
    for d in range(len(a)):
        result += a[d] * b[d]
    return result

# Trains word embeddings on CORPUS: learns BPE tokens, builds a fixed vocabulary, then
# trains a skip-gram model with negative sampling (see train_skip_gram).
class Embed:
    def __init__(self):
        self.tokenizer = BPETokenizer()
        self.token_freq_set: TokenFreqSet = self.tokenizer.train_bpe(" ".join(CORPUS).lower(), 500)
        self.vocabulary = Vocabulary(self.token_freq_set)

    def tokenize(self, text: str) -> TokenList:
        return self.tokenizer.tokenize(text.lower())

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

    # Rounds to the given number of decimal places, rounding halves up like JavaScript's
    # Math.round(value * 10**places) / 10**places. (Python's round() rounds halves to the
    # nearest even digit instead.)
    #
    # Example:  round_like_js(0.0000005, 6) -> 0.000001,  round_like_js(0.125, 2) -> 0.13
    def round_like_js(self, value: float, places: int) -> float:
        factor = 10 ** places
        return math.floor(value * factor + 0.5) / factor

    # How closely two vectors point the same way, ignoring their lengths:
    # 1 for the same direction, 0 for unrelated, -1 for opposite.
    # Returns 0 if either vector is all zeros, since it then has no direction.
    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        length_product = math.sqrt(dot(a, a)) * math.sqrt(dot(b, b))
        if length_product == 0:
            return 0.0
        return dot(a, b) / length_product

    # A word's embedding as a plain list, rounded to 6 decimal places
    def get_vector(self, network: SkipGramNetwork, word_index: int) -> list[float]:
        return [self.round_like_js(value, 6) for value in network.embedding(word_index)]

    # Compares the given vector with every token's embedding by cosine similarity, skipping the
    # tokens in exclude. Returns (token index, similarity) pairs in vocabulary order.
    # Each token's embedding is rounded to 6 places first (get_vector), as the TypeScript does;
    # the given vector is used as it is.
    #
    # Example, with 3 tokens and exclude = {1}:  [(0, 0.82), (2, -0.15)]
    def similarity_to_every_token(self, network: SkipGramNetwork, vector: list[float],
                                  exclude: set[int]) -> list[tuple[int, float]]:
        return [(index, self.cosine_similarity(vector, self.get_vector(network, index)))
                for index in range(network.vocab_size()) if index not in exclude]

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
        rand = mulberry32(42)

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

        negative_sampling_table = build_negative_sampling_table(self.vocabulary)
        network = SkipGramNetwork([token.token for token in self.vocabulary.index_to_token], embeded_vector_size, rand)
        trainer = SkipGramTrainer(network, positive_pairs, negative_sampling_table, rand, negative_samples)

        # How often (in epochs) to report progress: about 50 reports over the whole run,
        # but never less than every epoch. For example, 200 epochs -> every 4th epoch;
        # 30 epochs -> every epoch.
        step = max(1, epochs // 50)

        for epoch, loss in trainer.train(epochs):
            if epoch % step == 0 or epoch == epochs:
                yield EpochResult(epoch=epoch, loss=self.round_like_js(loss, 6))

        # Report on the query words that are single tokens; the rest only get a warning
        query_indices, warnings = self.find_query_indices(words)

        embeddings = [WordEmbedding(word=self.vocabulary.token_at(index).token, vector=self.get_vector(network, index))
                      for index in query_indices]

        # Nearest neighbors for each query word: the 5 other tokens with the most similar vectors
        neighbors: list[Neighbor] = []
        for query_index in query_indices:
            query_vector = self.get_vector(network, query_index)
            scores = [NeighborScore(word=self.vocabulary.token_at(index).token, score=self.round_like_js(similarity, 2))
                      for index, similarity in self.similarity_to_every_token(network, query_vector, {query_index})]
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
                    score=self.round_like_js(self.cosine_similarity(self.get_vector(network, query_indices[i]),
                                                                    self.get_vector(network, query_indices[j])), 2)
                ))

        # Analogies: for each "a - b + c", find the token whose vector is closest to
        # embedding(a) - embedding(b) + embedding(c), not counting a, b and c themselves
        analogies: list[Analogy] = []
        for a, b, c in ANALOGIES:
            a_index = self.vocabulary.token_to_index.get(Token(a))
            b_index = self.vocabulary.token_to_index.get(Token(b))
            c_index = self.vocabulary.token_to_index.get(Token(c))
            if a_index is None or b_index is None or c_index is None:
                continue

            # Uses the unrounded weights, as the TypeScript does
            a_vector, b_vector, c_vector = network.embedding(a_index), network.embedding(b_index), network.embedding(c_index)
            result_vector = [a_vector[d] - b_vector[d] + c_vector[d] for d in range(network.dim)]

            candidates = self.similarity_to_every_token(network, result_vector, {a_index, b_index, c_index})
            if candidates:
                # max() keeps the first of any tied scores, so a tie goes to the earlier token
                best_index, best_score = max(candidates, key=lambda candidate: candidate[1])
                analogies.append(Analogy(
                    query=f"{a} - {b} + {c}",
                    result=self.vocabulary.token_at(best_index).token,
                    score=self.round_like_js(best_score, 2)
                ))

        # Save the trained model, so it can be used later without retraining
        saved_model = {
            "type": "word2vec-skipgram",
            "dimensions": network.dim,
            "vocab": [token.token for token in self.vocabulary.index_to_token],
            "merges": [{"pair": [rule.pair[0].token, rule.pair[1].token], "merged": rule.merged.token}
                       for rule in self.tokenizer.merge_rules],
            "embeddings": {self.vocabulary.token_at(index).token: self.get_vector(network, index)
                           for index in range(network.vocab_size())},
        }
        weights_path = Path(__file__).resolve().parent.parent / ".data" / "embedding-weights.json"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_text(json.dumps(saved_model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        yield TrainEmbedResult(
            embeddings=embeddings,
            neightbors=neighbors,
            similarities=similarities,
            analogies=analogies,
            warnings=warnings
        )
