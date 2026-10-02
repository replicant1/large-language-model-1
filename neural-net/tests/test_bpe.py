import unittest
import re
from bpe import TokenList
from bpe import Token
from bpe import BPETokenizer

class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BPETokenizer()

    def test_count_words_short_string(self):
        text = "the cat sat!"
        word_counts = self.tokenizer.word_frequencies(text)
        print(f"output from bpe tokenizer = {word_counts}")  
        self.assertEqual(1, word_counts["the"])
        self.assertEqual(1, word_counts["cat"])
        self.assertEqual(1, word_counts["sat"])
        self.assertEqual(1, word_counts["!"])
        self.assertEqual(2, word_counts[" "])
        
    def test_count_words_medium_string(self):
        text = "the cat sat on the mat."
        word_counts = self.tokenizer.word_frequencies(text)
        print(f"output from bpe tokenizer = {word_counts}")  
        self.assertEqual(2, word_counts["the"])
        self.assertEqual(1, word_counts["cat"])
        self.assertEqual(1, word_counts["sat"])
        self.assertEqual(1, word_counts["on"])
        self.assertEqual(1, word_counts["mat"])
        self.assertEqual(1, word_counts["."])
        self.assertEqual(5, word_counts[" "])

    def test_count_words_empty_string(self):
        text = ""
        word_counts = self.tokenizer.word_frequencies(text)
        self.assertEqual(0, len(word_counts))
        
    def test_merge_character_pairs_to_tokens(self):
        input_tokens = ["c", "a", "t", "s"]
        char_pair = ("c", "a")
        when_merged = "ca" # the character pair "c" and "a" merge into "ca"
        merged_tokens = self.tokenizer.merge_pairs_to_tokens(input_tokens, char_pair, when_merged)
        print(f"output from merge_pairs_to_tokens = {merged_tokens}")
        self.assertEqual(["ca", "t", "s"], merged_tokens)
        
    def test_merge_pairs_to_tokens_round_1(self):
        input_tokens = ["a", "a", "a", "b", "d", "a", "a", "a", "b", "a", "c"]
        char_pair = ("a", "b")
        when_merged = "ab" # the character pair "a" and "b" do not exist in the input tokens
        merged_tokens = self.tokenizer.merge_pairs_to_tokens(input_tokens, char_pair, when_merged)
        print(f"output from merge_pairs_to_tokens = {merged_tokens}")
        self.assertEqual(["a", "a", "ab", "d", "a", "a", "ab", "a", "c"], merged_tokens)
        
    def test_merge_pairs_to_tokens_round_2(self):
        input_tokens = ["z", "a", "b", "d", "z", "a", "b", "a", "c"]
        char_pair = ("a", "b")
        when_merged = "ab" # the character pair "a" and "b" merge into "ab"
        merged_tokens = self.tokenizer.merge_pairs_to_tokens(input_tokens, char_pair, when_merged)
        print(f"output from merge_pairs_to_tokens = {merged_tokens}")
        self.assertEqual(["z", "ab", "d", "z", "ab", "a", "c"], merged_tokens)
        
    def test_merge_pairs_to_tokens_round_3(self):
        input_tokens = ["x", "y", "x", "y", "z"]
        char_pair = ("x", "y")
        when_merged = "xy" # the character pair "x" and "y" merge into "xy"
        merged_tokens = self.tokenizer.merge_pairs_to_tokens(input_tokens, char_pair, when_merged)
        print(f"output from merge_pairs_to_tokens = {merged_tokens}")
        self.assertEqual(["xy", "xy", "z"], merged_tokens)
        
    def test_merge_pairs_to_tokens_round_4(self):
        input_tokens = ["z", "y", "d", "z", "y", "a", "c"]
        char_pair = ("z", "y")
        when_merged = "zy" # the character pair "z" and "y" merge into "zy"
        merged_tokens = self.tokenizer.merge_pairs_to_tokens(input_tokens, char_pair, when_merged)
        print(f"output from merge_pairs_to_tokens = {merged_tokens}")
        self.assertEqual(["zy", "d", "zy", "a", "c"], merged_tokens)
        
    def test_train_bpe(self):
        tokens = TokenList()
        tokens.entries.append(Token("hug"))
        tokens.entries.append(Token("hugs"))
        tokens.entries.append(Token("bug"))
        tokens.entries.append(Token("bugs"))
        tokens = self.tokenizer.train_bpe(tokens)
        print(f"output from train_bpe = {tokens}")
        #self.assertIsInstance(tokens, list)
        
    def test_train_bpe_from_shakespeare(self):
        text = "Shall I compare thee to a summer's day?"
        word_freqs = self.tokenizer.word_frequencies(text)
        token_list = TokenList()
        for word in word_freqs:
            token_list.entries.append(Token(word))
        tokens = self.tokenizer.train_bpe(token_list)
        print(f"output from train_bpe_from_shakespeare = {tokens}")
        self.assertIsInstance(tokens, list)
        
    def test_train_bpe_from_shakespeare_again(self):
        text = "summer's more more"
        word_freqs = self.tokenizer.word_frequencies(text)
        token_list = TokenList()
        for word in word_freqs:
            token_list.entries.append(Token(word))
        tokens = self.tokenizer.train_bpe(token_list)
        #print(f"output from train_bpe_from_shakespeare = {tokens}")
        self.assertIsInstance(tokens, list)
        
    def _most_frequent_word(self, word_counts: dict[str, int]) -> str:
        if not word_counts:
            return ""
        return max(word_counts, key=word_counts.get)

if __name__ == "__main__":
    unittest.main()