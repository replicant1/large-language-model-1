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
        self.assertEqual(["ca", "t", "s"], [token.token for token in merged_token_list.entries])
        
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
        
    def test_tokenize(self):
        text = "fred fed ted bread, ted fed fred bread"
        result = self.tokenizer.train_bpe(text)
        print(f"output from train_bpe = {result}")
        expected_token_strings = [",", "a", "b", "br", "bre", "brea", "bread", "d", "e", "ed",
                                  "f", "fed", "fr", "fred", "r", "t", "ted"]
        for token_string in expected_token_strings:
            self.assertTrue(Token(token_string) in result.entries)
        self.assertEqual(len(expected_token_strings), len(result.entries))
        
    def test_tokenize_2(self):
        text = "Walked Talked Byzked"
        result = self.tokenizer.train_bpe(text)
        expected_token_strings = ["W", "a", "l", "k", "e", "d", "T", "B", "y", "z", "ke", "ked", "al", "alked"]
        print(f"output from train_bpe_2 = {result}")
        for token_string in expected_token_strings:
            self.assertTrue(Token(token_string) in result.entries)
        self.assertEqual(len(expected_token_strings), len(result.entries))

    # --- Pair counts must be weighted by word frequency ---

    def test_tokenize_repeated_word_is_merged(self):
        # "aa" occurs twice, so the pair (a, a) occurs twice and should be merged
        result = self.tokenizer.train_bpe("aa aa")
        print(f"output from tokenize = {result}")
        self.assertIn(Token("aa"), result.entries)
        self.assertEqual({"a", "aa"}, {token.token for token in result.entries})

    def test_count_pairs_weighted_by_word_frequency(self):
        hugTokenList = TokenList()
        for char in "hug":
            hugTokenList.add(Token(char))
        pugTokenList = TokenList()
        for char in "bug":
            pugTokenList.add(Token(char))
        hugsTokenList = TokenList()
        for char in "hugs":
            hugsTokenList.add(Token(char))
        bugsTokenList = TokenList()
        for char in "bugs":
            bugsTokenList.add(Token(char))
        word_to_token_list = {
            "hug": hugTokenList,
            "pug": pugTokenList,
            "hugs": hugsTokenList,
            "bugs": bugsTokenList,
        }
        word_freqs = {"hug": 1, "pug": 1, "hugs": 1, "bugs": 1}
        pair_frequencies = self.tokenizer.count_adjacent_token_pair_frequencies(word_to_token_list, word_freqs)
        print(f"pair_frequencies = {pair_frequencies}")
        self.assertEqual(4, pair_frequencies[(Token("u"), Token("g"))])
        self.assertEqual(2, pair_frequencies[(Token("h"), Token("u"))])
        self.assertEqual(2, pair_frequencies[(Token("b"), Token("u"))])
        self.assertEqual(2, pair_frequencies[(Token("g"), Token("s"))])

    # --- apply_merge_rule_to_token_list edge cases ---

    def test_merge_overlapping_pair_merges_left_to_right(self):
        input_tokens = self._token_list(["a", "a", "a"])
        merge_rule = MergeRule((Token("a"), Token("a")), Token("aa"))
        merged_token_list = self.tokenizer.apply_merge_rule_to_token_list(input_tokens, merge_rule)
        self.assertEqual(["aa", "a"], self._strings(merged_token_list))

    def test_merge_already_merged_tokens(self):
        input_tokens = self._token_list(["ab", "c", "d"])
        merge_rule = MergeRule((Token("ab"), Token("c")), Token("abc"))
        merged_token_list = self.tokenizer.apply_merge_rule_to_token_list(input_tokens, merge_rule)
        self.assertEqual(["abc", "d"], self._strings(merged_token_list))

    def test_merge_pair_in_reverse_order_is_not_merged(self):
        input_tokens = self._token_list(["b", "a"])
        merge_rule = MergeRule((Token("a"), Token("b")), Token("ab"))
        merged_token_list = self.tokenizer.apply_merge_rule_to_token_list(input_tokens, merge_rule)
        self.assertEqual(["b", "a"], self._strings(merged_token_list))

    def test_merge_empty_token_list(self):
        merge_rule = MergeRule((Token("a"), Token("b")), Token("ab"))
        merged_token_list = self.tokenizer.apply_merge_rule_to_token_list(TokenList(), merge_rule)
        self.assertEqual([], self._strings(merged_token_list))

    def test_merge_single_token_list(self):
        merge_rule = MergeRule((Token("a"), Token("b")), Token("ab"))
        merged_token_list = self.tokenizer.apply_merge_rule_to_token_list(self._token_list(["a"]), merge_rule)
        self.assertEqual(["a"], self._strings(merged_token_list))

    def test_merge_does_not_modify_input_token_list(self):
        input_tokens = self._token_list(["a", "b", "c"])
        merge_rule = MergeRule((Token("a"), Token("b")), Token("ab"))
        merged_token_list = self.tokenizer.apply_merge_rule_to_token_list(input_tokens, merge_rule)
        self.assertEqual(["ab", "c"], self._strings(merged_token_list))
        self.assertEqual(["a", "b", "c"], self._strings(input_tokens))

    def _token_list(self, token_strings: list[str]) -> TokenList:
        token_list = TokenList()
        for token_string in token_strings:
            token_list.add(Token(token_string))
        return token_list

    def _strings(self, token_list: TokenList) -> list[str]:
        return [token.token for token in token_list.entries]

if __name__ == "__main__":
    unittest.main()