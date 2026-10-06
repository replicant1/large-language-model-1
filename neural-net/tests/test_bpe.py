import unittest
import re
from bpe import TokenList
from bpe import Token
from bpe import BPETokenizer
from bpe import MergeRule

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
        
    def test_count_words_medium_string(self):
        text = "the cat sat on the mat."
        word_counts = self.tokenizer.word_frequencies(text)
        print(f"output from bpe tokenizer = {word_counts}")  
        self.assertEqual(2, word_counts["the"])
        self.assertEqual(1, word_counts["cat"])
        self.assertEqual(1, word_counts["sat"])
        self.assertEqual(1, word_counts["on"])
        self.assertEqual(1, word_counts["mat"])

    def test_count_words_empty_string(self):
        text = ""
        word_counts = self.tokenizer.word_frequencies(text)
        self.assertEqual(0, len(word_counts))
        
    def test_merge_character_pairs_to_tokens(self):
        input_tokens = TokenList()
        input_tokens.add(Token("c"))
        input_tokens.add(Token("a"))
        input_tokens.add(Token("t"))
        input_tokens.add(Token("s"))
        char_pair = (Token("c"), Token("a"))
        merge_rule = MergeRule(char_pair, Token("ca"))
        merged_token_list = self.tokenizer.apply_merge_rule_to_token_list(input_tokens, merge_rule)
        print(f"output from merge_token_pair_in_token_list = {merged_token_list}")
        self.assertEqual("ca", merged_token_list.entries[0].token)
        
    def test_merge_pairs_to_tokens_round_1(self):
        input_tokens = TokenList()
        for char in ["a", "a", "a", "b", "d", "a", "a", "a", "b", "a", "c"]:
            input_tokens.add(Token(char))
        char_pair = (Token("a"), Token("b"))
        merge_rule = MergeRule(char_pair, Token("ab"))
        merged_token_list = self.tokenizer.apply_merge_rule_to_token_list(input_tokens, merge_rule)
        print(f"output from merge_token_pair_in_token_list = {merged_token_list}")
        self.assertEqual(["a", "a", "ab", "d", "a", "a", "ab", "a", "c"], [token.token for token in merged_token_list.entries])
        
    def test_merge_pairs_to_tokens_round_2(self):
        input_tokens = TokenList()
        for char in ["z", "a", "b", "d", "z", "a", "b", "a", "c"]:
            input_tokens.add(Token(char))
        char_pair = (Token("a"), Token("b"))
        merge_rule = MergeRule(char_pair, Token("ab"))
        merged_token_list = self.tokenizer.apply_merge_rule_to_token_list(input_tokens, merge_rule)
        print(f"output from merge_token_pair_in_token_list = {merged_token_list}")
        self.assertEqual(["z", "ab", "d", "z", "ab", "a", "c"], [token.token for token in merged_token_list.entries])
        
    def test_merge_pairs_to_tokens_round_3(self):
        input_tokens = TokenList()
        for char in ["x", "y", "x", "y", "z"]:
            input_tokens.add(Token(char))
        char_pair = (Token("x"), Token("y"))
        merge_rule = MergeRule(char_pair, Token("xy"))
        merged_token_list = self.tokenizer.apply_merge_rule_to_token_list(input_tokens, merge_rule)
        print(f"output from merge_token_pair_in_token_list = {merged_token_list}")
        self.assertEqual(["xy", "xy", "z"], [token.token for token in merged_token_list.entries])
        
    def test_merge_pairs_to_tokens_round_4(self):
        input_tokens = TokenList()
        for char in ["z", "y", "d", "z", "y", "a", "c"]:
            input_tokens.add(Token(char))
        char_pair = (Token("z"), Token("y"))
        merge_rule = MergeRule(char_pair, Token("zy"))
        merged_token_list = self.tokenizer.apply_merge_rule_to_token_list(input_tokens, merge_rule)
        print(f"output from merge_token_pair_in_token_list = {merged_token_list}")
        self.assertEqual(["zy", "d", "zy", "a", "c"], [token.token for token in merged_token_list.entries])
        
    def test_train_bpe(self):
        text = "fred fed ted bread, ted fed fred bread"
        pair_frequencies = self.tokenizer.word_frequencies(text)
        result = self.tokenizer.train_bpe(pair_frequencies)
        print(f"output from train_bpe = {result}")
        expected_token_strings = ["a", "t", "r", "e", "f", ",", "ed", "d", "b"]
        for token_string in expected_token_strings:
            self.assertTrue(Token(token_string) in result.entries)
        self.assertEqual(9, len(result.entries))
        
    def test_train_bpe_2(self):
        text = "Walked Talked Byzked"
        pair_frequencies = self.tokenizer.word_frequencies(text)
        tokens = self.tokenizer.train_bpe(pair_frequencies)
        print(f"output from train_bpe_2 = {tokens}")
        
    # def test_train_bpe_from_shakespeare(self):
    #     text = "Shall I compare thee to a summer's day?"
    #     pair_frequences = self.tokenizer.word_frequencies(text)
    #     print(f"output from train_bpe_from_shakespeare = {pair_frequences}")  
        
    # def test_train_bpe_from_shakespeare_again(self):
    #     text = "summer's more more"
    #     word_freqs = self.tokenizer.word_frequencies(text)
    #     token_list = TokenList()
    #     for word in word_freqs:
    #         token_list.entries.append(Token(word))
    #     tokens = self.tokenizer.train_bpe(token_list)
    #     #print(f"output from train_bpe_from_shakespeare = {tokens}")
    #     self.assertIsInstance(tokens, list)
        
    # def _most_frequent_word(self, word_counts: dict[str, int]) -> str:
    #     if not word_counts:
    #         return ""
    #     return max(word_counts, key=word_counts.get)

if __name__ == "__main__":
    unittest.main()