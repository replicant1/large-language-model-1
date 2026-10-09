from __future__ import annotations
import re
from dataclasses import dataclass
from collections import defaultdict

@dataclass(frozen=True)
class MergeRule:
    pair: tuple[Token, Token]
    merged: Token
    
@dataclass(frozen=True)
class Token:
    token: str
    def concat(self, other: Token) -> str:
        return self.token + other.token
    
@dataclass(frozen=True)
class TokenFreq:
    token: Token
    frequency: int
    
class TokenList:
    def __init__(self):
        self.entries = []
    
    def entries(self) -> list[Token]:
        return self.entries
    
    def add(self, entry: Token):
        self.entries.append(entry)
    
    def __str__(self) -> str:
        return ", ".join(entry.token for entry in self.entries)
    
    def __repr__(self) -> str:
        return f"[{self.__str__()}]"
    
    def size(self) -> int:
        return len(self.entries)
    
    def print(self):
        for index, entry in enumerate(self.entries):
            print(f"Token[{index}]: {entry.token}")
      
# List of frequencies with which each token occurs in a corpus, sorted from most 
# frequent to least frequent. Each token appears only once.      
class TokenFreqSet:
    def __init__(self):
        self.entries = set()
        
    def entries(self) -> set[TokenFreq]:
        return self.entries

    # Add a new token frequency to the set, combining it with any existing frequency for the same token.
    def add(self, new_token_freq: TokenFreq):
        old_freq = 0
        if (self.contains(new_token_freq.token.token)):
            old_freq = next(entry.frequency for entry in self.entries if entry.token == new_token_freq.token)
            self.entries.remove(next(entry for entry in self.entries if entry.token == new_token_freq.token))
        self.entries.add(TokenFreq(new_token_freq.token, old_freq + new_token_freq.frequency))

    def contains(self, token : str) -> bool:
        return any(entry.token.token == token for entry in self.entries)

    def size(self) -> int:
        return len(self.entries)
    
    def print(self):
        for entry in self.entries:
            print(f"Token: {entry.token} Frequency: {entry.frequency}")
            
    def __str__(self) -> str:
        return self.__repr__()

    # Returns the entries as a list, most frequent first, with ties in token order,
    # so the order is the same every run
    def sorted_by_frequency(self) -> list[TokenFreq]:
        return sorted(self.entries, key=lambda entry: (-entry.frequency, entry.token.token))

    def __repr__(self) -> str:
        return "[" + ", ".join(f"{entry.token.token}: {entry.frequency}" for entry in self.sorted_by_frequency()) + "]"

class BPETokenizer:
    def __init__(self):
        self.vocabulary = TokenFreqSet()
        self.merge_rules : list[MergeRule] = []

    # Split text into words and punctuation, in order and keeping repeats; whitespace is dropped
    def split_into_words(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text)

    # Prepare a table that maps each word to its count in the given text
    # text: string to count words from
    # returns a dictionary with words as keys and their counts as values
    def word_frequencies(self, text: str) -> dict[str, int]:
        word_counts = dict()
        words = self.split_into_words(text)
        for word in words:
            current_count = word_counts.get(word, 0)
            word_counts[word] = current_count + 1
        return word_counts
    
    # Count how often each adjacent token pair occurs, weighting each word's pairs by how often the word occurs
    def count_adjacent_token_pair_frequencies(self, word_to_token_list: dict[str, TokenList], word_freqs: dict[str, int]) -> dict[tuple[Token, Token], int]:
        pairs = defaultdict(int)
        for word, tokens in word_to_token_list.items():
            for index in range(len(tokens.entries) - 1):
                this_token : Token = tokens.entries[index]
                next_token : Token = tokens.entries[index + 1]
                this_pair = (this_token, next_token)    
                pairs[this_pair] += word_freqs[word]
                
        print("--- ADJACENT TOKEN PAIR COUNTS ---")
        for pair, count in pairs.items():
            print(f"{pair} occurs {count} times")
        print("--- END OF ADJACENT TOKEN PAIR COUNTS ---")
        return pairs
    
    def apply_merge_rule_to_token_list(self, input_tokens: TokenList, merge_rule: MergeRule) -> TokenList:
        print(f"Applying merge rule: {merge_rule} to input_tokens: {input_tokens.entries}")
        result = TokenList()
        i = 0
        while i < len(input_tokens.entries):
            if i < len(input_tokens.entries) - 1 and (input_tokens.entries[i], input_tokens.entries[i + 1]) == merge_rule.pair:
                merged_token = merge_rule.merged
                result.entries.append(merged_token)
                i += 2
            else:
                result.entries.append(input_tokens.entries[i])
                i += 1
        print(f"Result after applying merge rule: {result.entries}\n")
        return result   
    
    # Applies a merge rule to every word's token list.
    # Returns a new dictionary; the input dictionary and its token lists are left unchanged.
    def apply_merge_rule_to_token_map(self, word_to_token_list: dict[str, TokenList], merge_rule: MergeRule) -> dict[str, TokenList]:
        merged_word_to_token_list = dict[str, TokenList]()
        for word, token_list in word_to_token_list.items():
            print(f"Applying merge rule to TokenList for word: {word}")
            merged_word_to_token_list[word] = self.apply_merge_rule_to_token_list(token_list, merge_rule)
        return merged_word_to_token_list
    
    # You must have already called train_bpe before using this method to tokenize text.
    # Splits text into words the same way training does, then applies the learned
    # merge rules to each word in the order they were learned.
    def tokenize(self, text: str) -> TokenList:
        result = TokenList()
        for word in self.split_into_words(text):
            word_tokens = TokenList()
            for char in word:
                word_tokens.add(Token(char))
            for merge_rule in self.merge_rules:
                word_tokens = self.apply_merge_rule_to_token_list(word_tokens, merge_rule)
            for token in word_tokens.entries:
                result.add(token)
        return result
        
    # Train the BPE tokenizer by repeatedly merging the most frequent pairs of characters in the input text.
    # Keep merging the most frequent pairs until reaching the maximum number of merges or no more merges are possible.
    # returns a set of token frequencies (vocabulary) after applying BPE merges
    def train_bpe(self, corpus: str, max_merges: int = 1000) -> TokenFreqSet:
        print(f"Into train_bpe with corpus of length {len(corpus)}")
        
        self.merge_rules = []
        word_freqs = self.word_frequencies(corpus)
        
        # Convert each word into a list of Token objects and store them in tokens_per_word dictionary
        # This approach means that the spaces betwen words are not represented as tokens
        vocabulary = TokenFreqSet()
        
        word_to_token_list = dict[str, TokenList]()
        for word in word_freqs:
            print(f"initial processing identifies word: {word}")
            word_to_token_list[word] = TokenList()
            for char in word:
                word_to_token_list[word].add(Token(char)) 
                vocabulary.add(TokenFreq(Token(char), word_freqs[word]))
                
        print(f"INITIAL VOCABULARY: {vocabulary}")
        
        for step in range(max_merges):
            print(f"--- STEP {step} ---")
            
            token_pair_frequencies = self.count_adjacent_token_pair_frequencies(word_to_token_list, word_freqs)
            if not token_pair_frequencies:
                break
            
            most_freq_token_pair = max(token_pair_frequencies, key=token_pair_frequencies.get)
            print(f"most frequent token pair: {most_freq_token_pair} -> {token_pair_frequencies[most_freq_token_pair]}")
            
            if token_pair_frequencies[most_freq_token_pair] < 2:
                            break
        
            # Perform the token merge for the most frequently occuring token pair
            merge_rule = MergeRule(most_freq_token_pair, Token(''.join([token.token for token in most_freq_token_pair])))
            word_to_token_list = self.apply_merge_rule_to_token_map(word_to_token_list, merge_rule)
            
            self.merge_rules.append(merge_rule)
            
            vocabulary.add(TokenFreq(merge_rule.merged, token_pair_frequencies[most_freq_token_pair]))
            
            print(f"Merged vocabulary: {vocabulary}")
            
        print(f"FINAL VOCABULARY: {vocabulary}")
        
        self.vocabulary = vocabulary
        return vocabulary
      
        