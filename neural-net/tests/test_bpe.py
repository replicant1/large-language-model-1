import unittest
import re
from bpe import TokenList
from bpe import Token
from bpe import BPETokenizer
from bpe import MergeRule
from bpe import TokenFreq
from bpe import TokenFreqSet
from bpe import Vocabulary

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
        
    def test_tokenize_1(self):
        text = "fred fed ted bread, ted fed fred bread"
        result = self.tokenizer.train_bpe(text)
        print(f"output from train_bpe = {result}")
        expected_token_freqs = self._token_freqs({
            "d": 8, "e": 8, "ed": 6, "f": 4, "r": 4,
            "a": 2, "b": 2, "br": 2, "bre": 2, "brea": 2, "bread": 2,
            "fed": 2, "fr": 2, "fred": 2, "t": 2, "ted": 2, ",": 1,
        })
        self.assertEqual(expected_token_freqs, result.entries)
        
    def test_tokenize_2(self):
        text = "Walked Talked Byzked"
        result = self.tokenizer.train_bpe(text)
        expected_token_freqs = self._token_freqs({
            "d": 3, "e": 3, "k": 3, "ke": 3, "ked": 3,
            "a": 2, "al": 2, "alked": 2, "l": 2,
            "B": 1, "T": 1, "W": 1, "y": 1, "z": 1,
        })
        print(f"output from train_bpe_2 = {result}")
        self.assertEqual(expected_token_freqs, result.entries)

    # --- Pair counts must be weighted by word frequency ---

    def test_tokenize_repeated_word_is_merged(self):
        # "aa" occurs twice, so the pair (a, a) occurs twice and should be merged
        result = self.tokenizer.train_bpe("aa aa")
        print(f"output from tokenize = {result}")
        self.assertIn(TokenFreq(Token("aa"), 2), result.entries)
        self.assertEqual(self._token_freqs({"a": 4, "aa": 2}), result.entries)

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

    # --- apply_merge_rule_to_token_map ---

    def test_merge_map_applies_rule_to_every_word(self):
        word_to_token_list = {
            "fed": self._token_list(["f", "e", "d"]),
            "ted": self._token_list(["t", "e", "d"]),
            "bread": self._token_list(["b", "r", "e", "a", "d"]),
        }
        merge_rule = MergeRule((Token("e"), Token("d")), Token("ed"))
        merged = self.tokenizer.apply_merge_rule_to_token_map(word_to_token_list, merge_rule)
        self.assertEqual({
            "fed": ["f", "ed"],
            "ted": ["t", "ed"],
            "bread": ["b", "r", "e", "a", "d"],
        }, {word: self._strings(token_list) for word, token_list in merged.items()})

    def test_merge_map_does_not_modify_input(self):
        word_to_token_list = {
            "fed": self._token_list(["f", "e", "d"]),
            "ted": self._token_list(["t", "e", "d"]),
        }
        original_fed_tokens = word_to_token_list["fed"]
        merge_rule = MergeRule((Token("e"), Token("d")), Token("ed"))
        merged = self.tokenizer.apply_merge_rule_to_token_map(word_to_token_list, merge_rule)
        self.assertIsNot(word_to_token_list, merged)
        self.assertIs(original_fed_tokens, word_to_token_list["fed"])
        self.assertEqual({
            "fed": ["f", "e", "d"],
            "ted": ["t", "e", "d"],
        }, {word: self._strings(token_list) for word, token_list in word_to_token_list.items()})

    def _token_list(self, token_strings: list[str]) -> TokenList:
        token_list = TokenList()
        for token_string in token_strings:
            token_list.add(Token(token_string))
        return token_list

    def _token_freqs(self, token_string_to_freq: dict[str, int]) -> set[TokenFreq]:
        return {TokenFreq(Token(token_string), freq) for token_string, freq in token_string_to_freq.items()}

    def _strings(self, token_list: TokenList) -> list[str]:
        return [token.token for token in token_list.entries]

class TestVocabulary(unittest.TestCase):
    def setUp(self):
        # "ed" is most frequent; "a" and "b" tie, so they are in token order
        token_freqs = TokenFreqSet()
        for token_string, freq in [("b", 2), ("ed", 6), ("a", 2), ("z", 1)]:
            token_freqs.add(TokenFreq(Token(token_string), freq))
        self.vocabulary = Vocabulary(token_freqs)

    def test_indexes_are_by_frequency_then_token(self):
        self.assertEqual(["ed", "a", "b", "z"], [token.token for token in self.vocabulary.index_to_token])

    def test_size(self):
        self.assertEqual(4, self.vocabulary.size())

    def test_index_of_and_token_at_are_inverses(self):
        self.assertEqual(0, self.vocabulary.index_of(Token("ed")))
        self.assertEqual(Token("b"), self.vocabulary.token_at(2))
        for index in range(self.vocabulary.size()):
            self.assertEqual(index, self.vocabulary.index_of(self.vocabulary.token_at(index)))

    def test_index_of_unknown_token_raises(self):
        with self.assertRaises(KeyError):
            self.vocabulary.index_of(Token("q"))

    def test_contains(self):
        self.assertTrue(self.vocabulary.contains(Token("ed")))
        self.assertFalse(self.vocabulary.contains(Token("q")))

    def test_frequency_at(self):
        self.assertEqual(6, self.vocabulary.frequency_at(0))
        self.assertEqual(1, self.vocabulary.frequency_at(3))

    def test_encode_keeps_order_and_repeats(self):
        tokens = TokenList()
        for token_string in ["b", "ed", "b", "z"]:
            tokens.add(Token(token_string))
        self.assertEqual([2, 0, 2, 3], self.vocabulary.encode(tokens))

    def test_encode_drops_unknown_tokens(self):
        tokens = TokenList()
        for token_string in ["a", "q", "ed"]:
            tokens.add(Token(token_string))
        self.assertEqual([1, 0], self.vocabulary.encode(tokens))

    def test_decode_reverses_encode(self):
        tokens = TokenList()
        for token_string in ["z", "a", "ed", "a"]:
            tokens.add(Token(token_string))
        decoded = self.vocabulary.decode(self.vocabulary.encode(tokens))
        self.assertEqual(["z", "a", "ed", "a"], [token.token for token in decoded.entries])

    def test_vocabulary_from_trained_tokenizer(self):
        tokenizer = BPETokenizer()
        vocabulary = Vocabulary(tokenizer.train_bpe("fred fed ted bread, ted fed fred bread"))
        self.assertEqual(17, vocabulary.size())
        self.assertEqual([Token("d"), Token("e"), Token("ed")], vocabulary.index_to_token[:3])
        encoded = vocabulary.encode(tokenizer.tokenize("fed red"))
        self.assertEqual(["fed", "r", "ed"], [token.token for token in vocabulary.decode(encoded).entries])

if __name__ == "__main__":
    unittest.main()