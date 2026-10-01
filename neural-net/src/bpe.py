import re

re.search(r"\w+|\s|[^\w\s]", "some string to search")

class BPETokenizer:
    def __init__(self):
        pass

    # Prepare a table that maps each word to its count in the given text
    # text: string to count words from
    # returns a dictionary with words as keys and their counts as values
    def count_words(self, text: str) -> dict[str, int]:
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
    def merge_character_pairs_to_tokens(self, input_tokens: list[str], char_pair: tuple[str, str], when_merged: str) -> list[str]:
        merged_tokens = []
        i = 0
        while i < len(input_tokens):
            if i < len(input_tokens) - 1 and (input_tokens[i], input_tokens[i + 1]) == char_pair:
                merged_tokens.append(when_merged)
                i += 2
            else:
                merged_tokens.append(input_tokens[i])
                i += 1
        return merged_tokens
    