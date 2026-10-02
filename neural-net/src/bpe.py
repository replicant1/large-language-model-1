from __future__ import annotations
import re
from dataclasses import dataclass
from collections import defaultdict

# re.search(r"\w+|\s|[^\w\s]", "some string to search")

@dataclass(frozen=True)
class Merge:
    pair: tuple[str, str]
    merged: str
    frequency: int
    
@dataclass(frozen=True)
class Token:
    token: str
    #frequency: int
    def concat(self, other: Token) -> str:
        return self.token + other.token
    
# @dataclass(frozen=True)
# class VocabEntry:
#     token: str
#     frequency: int
    
class TokenList:
    def __init__(self):
        self.entries = []
    
    def entries(self) -> list[Token]:
        return self.entries
    
    def add(self, entry: Token):
        self.entries.append(entry)

    def merge(self, entry: Token):
        old_entry = self._find_entry_by_str(entry.token)
        if old_entry is None:
            self.entries.append(entry)
        else:
            self.entries.remove(old_entry)
            self.entries.append(Token(entry.token))
            
    def find_entry_by_token(self, token: str) -> Token | None:
        return self._find_entry_by_str(token)
        
    def _find_entry_by_str(self, token: str) -> Token | None:
        for entry in self.entries:
            if entry.token == token:
                return entry
        return None
    
    def size(self) -> int:
        return len(self.entries)
    
    def print(self):
        for index, entry in enumerate(self.entries):
            print(f"Token[{index}]: {entry.token}")
    
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
    
    def count_adjacent_token_pair_frequencies(self, tokens : TokenList) -> dict[tuple[Token, Token], int]:
        pairs = defaultdict(int)
        print(f"Counting adjacent token pair frequencies for #tokens: {len(tokens.entries)}, tokens: {tokens}")
        for index in range(len(tokens.entries) - 1):
            this_token : Token = tokens.entries[index]
            next_token : Token = tokens.entries[index + 1]
            print(f"this_token =  {this_token}, Next token =  {next_token}")
            this_pair = (this_token, next_token)    
            pairs[this_pair] += 1
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
    
    def count_adjacent_letter_pair_frequencies(self, text: str) -> dict[str, int]:
        pairs = defaultdict(int)
        for i in range(len(text) - 1):
            pairs[(text[i], text[i + 1])] += 1
        return pairs
        
    # Train the BPE tokenizer by repeatedly merging the most frequent pairs of characters in the input text.
    # Keep merging the most frequent pairs until reaching the maximum number of merges or no more merges are possible.
    # word_freqs: dictionary with words as keys and their counts as values
    # max_merges: maximum number of merges to perform
    # returns a list of tokens after applying BPE merges
    def train_bpe(self, tokens: TokenList, max_merges: int = 1000) -> list[str]:
        print(f"tokens at input:")
        tokens.print()
        
        for step in range(1):
            print(f"--- STEP {step} ---")
            pair_frequencies = self.count_adjacent_token_pair_frequencies(tokens)
            print(f"pair frequencies: {pair_frequencies}")
            if not pair_frequencies:
                break
            # print(f"pair frequencies before removing: {pair_frequencies}")
            # pairs_not_counted = {pair: freq for pair, freq in pair_frequencies.items() if not tokens.find_entry_by_token(''.join(pair))}
            # print(f"pair frequencies after removing: {pairs_not_counted}")
            
            # best = max(pairs_not_counted, key=pairs_not_counted.get)
            # print(f"most frequent pair: {best} -> {pairs_not_counted[best]}")
        
            # tokens.merge(Token(''.join(best), 0))
            # print(f"vocabulary after merging most frequent pair:")
            # tokens.print()
            
        print("--- END STATE VOCABULARY ---")
        tokens.print()
        
        # vocab = self.merge_vocab(vocab, best)
        # #merges.append(best)
        # print("vocab after first merge ")
        # vocab.print_entries()
        
        # for i in range(max_merges):
        #     print(f"i == {i}")
        #     pairs = self.count_adjacent_pair_frequencies(vocab)  
        #     if not pairs:
        #         break
            
        #     # find the most frequently occurrring pair
        #     best = max(pairs, key = pairs.get)
        #     print(f"MOST FREQUENTLY OCCURRING PAIR: {best}, frequency: {pairs[best]}")
        #     vocab = self.merge_vocab(vocab, best)
        #     merges.append(best)
        #     print(f"vocab after merge ({len(vocab)} tokens): {vocab}")
        #     print(f'Merge {i+1}: {best} -> Result: {list(vocab.keys())}')
                
        # print(f'\nLearned Merges ({len(merges)}): {merges}')
        #return list(vocab.keys())
        