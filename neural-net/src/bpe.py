from __future__ import annotations
import re
from dataclasses import dataclass
from collections import defaultdict

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
    
    def __str__(self) -> str:
        return ", ".join(entry.token for entry in self.entries)
    
    def __repr__(self) -> str:
        return f"[{self.__str__()}]"
    
    def size(self) -> int:
        return len(self.entries)
    
    def print(self):
        for index, entry in enumerate(self.entries):
            print(f"Token[{index}]: {entry.token}")
            
class TokenSet:
    def __init__(self):
        self.entries = set()
        
    def entries(self) -> set[Token]:
        return self.entries

    def add(self, entry: Token):
        self.entries.add(entry)

    def contains(self, token : str) -> bool:
        return any(entry.token == token for entry in self.entries)

    def size(self) -> int:
        return len(self.entries)
    
    def print(self):
        for entry in self.entries:
            print(f"Token: {entry.token}")
    
class BPETokenizer:
    def __init__(self):
        pass

    # Prepare a table that maps each word to its count in the given text
    # text: string to count words from
    # returns a dictionary with words as keys and their counts as values
    def word_frequencies(self, text: str) -> dict[str, int]:
        word_counts = dict()
        words = re.findall(r"\w+|[^\w\s]", text)
        for word in words:
            current_count = word_counts.get(word, 0)
            word_counts[word] = current_count + 1
        return word_counts
    
    def count_adjacent_token_pair_frequencies(self, word_to_token_list: dict[str, TokenList]) -> dict[tuple[Token, Token], int]:
        pairs = defaultdict(int)
        for word, tokens in word_to_token_list.items():
            for index in range(len(tokens.entries) - 1):
                this_token : Token = tokens.entries[index]
                next_token : Token = tokens.entries[index + 1]
                this_pair = (this_token, next_token)    
                pairs[this_pair] += 1
                
        print("--- TOKEN PAIR COUNTS ---")
        for pair, count in pairs.items():
            print(f"Token pair {pair} occurs {count} times")
        print("--- END OF TOKEN PAIR COUNTS ---")
        return pairs
    
    def merge_token_pair_in_token_list(self, input_tokens: TokenList, token_pair: tuple[Token, Token]) -> TokenList:
        result = TokenList()
        i = 0
        while i < len(input_tokens.entries):
            if i < len(input_tokens.entries) - 1 and (input_tokens.entries[i], input_tokens.entries[i + 1]) == token_pair:
                merged_token = Token(''.join([token.token for token in token_pair]))
                result.entries.append(merged_token)
                i += 2
            else:
                result.entries.append(input_tokens.entries[i])
                i += 1
        return result   
    
    # Merges the most frequent pair in all words in the vocabulary
    def merge_token_pair_in_token_list_map(self, word_to_token_list: dict[str, TokenList], token_pair: tuple[Token, Token]):
        print(f"Starting merge for token pair: {token_pair}")
        pre_merge = ' '.join([token.token for token in token_pair])
        post_merge = ''.join([token.token for token in token_pair])
        print(f"Merging pair: {token_pair} -> {post_merge}")
        
        # Perform the merge in the input word_to_token_list dictionary AND in the vocabulary output dictionary
        for word, token_list in word_to_token_list.items():
            new_word = word.replace(pre_merge, post_merge)
            word_to_token_list[word] = self.merge_token_pair_in_token_list(token_list, token_pair)
            print(f"Merged token list for {new_word}: {word_to_token_list[word]}")
    
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
    def train_bpe(self, word_freqs: dict[str, int], max_merges: int = 1000):
        print(f"word_freqs at input: {word_freqs}")
        
        # Convert each word into a list of Token objects and store them in tokens_per_word dictionary
        word_to_token_list = dict[str, TokenList]()
        for word in word_freqs:
            print(f"processing word: {word}")
            word_to_token_list[word] = TokenList()
            for char in word:
                word_to_token_list[word].add(Token(char)) 
                
        print(f"tokens_per_word after initial tokenization: {word_to_token_list}")
        
        for step in range(max_merges):
            print(f"--- STEP {step} ---")
            token_pair_frequencies = self.count_adjacent_token_pair_frequencies(word_to_token_list)
            if not token_pair_frequencies:
                break
            
            most_freq_token_pair = max(token_pair_frequencies, key=token_pair_frequencies.get)
            print(f"most frequent pair: {most_freq_token_pair} -> {token_pair_frequencies[most_freq_token_pair]}")
            
            if token_pair_frequencies[most_freq_token_pair] < 2:
                break
        
            # Perform the token merge for the most frequently occuring token pair
            self.merge_token_pair_in_token_list_map(word_to_token_list, most_freq_token_pair)
            
            merged = word_to_token_list
            print(f"merged =  {merged}")
            
            
        print("--- END STATE VOCABULARY ---")
      
        