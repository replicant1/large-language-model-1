import re
from dataclasses import dataclass
from collections import defaultdict

re.search(r"\w+|\s|[^\w\s]", "some string to search")

@dataclass(frozen=True)
class Merge:
    pair: tuple[str, str]
    merged: str
    frequency: int
        

class BPETokenizer:
    def __init__(self):
        pass

    # Prepare a table that maps each word to its count in the given text
    # text: string to count words from
    # returns a dictionary with words as keys and their counts as values
    def word_frequencies(self, text: str) -> dict[str, int]:
        word_counts = dict()
        words = re.findall(r"\w+|\s|[^\w\s]", text)
        print(f"num words = {len(words)}")
        for word in words:
            current_count = word_counts.get(word, 0)
            word_counts[word] = current_count + 1
        return word_counts
    
    # This is where the name "byte pair encoding" comes from: it merges the most frequent pairs of bytes (or characters) 
    # in a sequence.
    # input_tokens: list of tokens to process
    # char_pair: tuple of two characters to merge
    # when_merged: string to replace the merged character pair with
    # returns a new list of tokens with the specified character pair merged
    def merge_pairs_to_tokens(self, input_tokens: list[str], pair: tuple[str, str], when_merged: str) -> list[str]:
        merged_tokens = []
        i = 0
        while i < len(input_tokens):
            if i < len(input_tokens) - 1 and (input_tokens[i], input_tokens[i + 1]) == pair:
                merged_tokens.append(when_merged)
                i += 2
            else:
                merged_tokens.append(input_tokens[i])
                i += 1
        return merged_tokens
    
    def count_adjacent_pair_frequencies(self, vocab: dict[str, int]) -> dict[tuple[str, str], int]:
        pairs = defaultdict(int)
        for vocab_term, vocab_freq in vocab.items():
            vocab_sub_term = vocab_term.split() # split according to any whitespace, and discard empty strings from the result
            for i in range(len(vocab_sub_term) - 1):
                pairs[(vocab_sub_term[i], vocab_sub_term[i + 1])] += vocab_freq
                print(f"pair: {(vocab_sub_term[i], vocab_sub_term[i + 1])}, frequency incremented to {pairs[(vocab_sub_term[i], vocab_sub_term[i + 1])]}")
        return pairs
    
    # Merges the most frequent pair in all words in the vocabulary
    def merge_vocab(self, vocab_in: dict[str, int], pair: tuple[str, str]) -> dict[str, int]:
        vocab_out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in vocab_in.items():
            # Replace the bigram with the merged replacement in the word
            new_word = word.replace(bigram, replacement)
            vocab_out[new_word] = freq
        return vocab_out
        
    # Train the BPE tokenizer by repeatedly merging the most frequent pairs of characters in the input text.
    # Keep merging the most frequent pairs until reaching the maximum number of merges or no more merges are possible.
    # word_freqs: dictionary with words as keys and their counts as values
    # max_merges: maximum number of merges to perform
    # returns a list of tokens after applying BPE merges
    def train_bpe(self, word_freqs: dict[str, int], max_merges: int = 1000) -> list[str]:
        print(f"word_freqs: {word_freqs}")
        vocab = dict[str, int]()
        for word in word_freqs:
            spaced = " ".join(list(word) + ["</w>"])
            vocab[spaced] = word_freqs[word]
        print(f"vocab: {vocab}")
        
        num_merges = 3
        merges = []
        
        for i in range(num_merges):
            print(f"i == {i}")
            pairs = self.count_adjacent_pair_frequencies(vocab)  
            if not pairs:
                break
            
            # find the most frequently occurrring pair
            best = max(pairs, key = pairs.get)
            print(f"MOST FREQUENTLY OCCURRING PAIR: {best}, frequency: {pairs[best]}")
            vocab = self.merge_vocab(vocab, best)
            merges.append(best)
            print(f"vocab after merge: {vocab}")
            print(f'Merge {i+1}: {best} -> Result: {list(vocab.keys())}')
                
        print('\nLearned Merges:', merges)
        return list(vocab.keys())
        