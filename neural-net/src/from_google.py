from collections import Counter, defaultdict


def get_stats(vocab):
  """Counts frequencies of adjacent pairs in the vocabulary."""
  pairs = defaultdict(int)
  for word, freq in vocab.items():
    symbols = word.split()
    for i in range(len(symbols) - 1):
      pairs[symbols[i], symbols[i + 1]] += freq
  return pairs


def merge_vocab(pair, v_in):
    print(f'Merging pair: {pair} into vocabulary')
    """Merges the most frequent pair in all words in the vocabulary."""
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in v_in.items():
        # Replace the split pair with the merged single token
        new_word = word.replace(bigram, replacement)
        v_out[new_word] = freq
    print(f'  vocabulary after merging:    ')
    for word, freq in v_out.items():
        print(f'    {word}: {freq}')
    return v_out


# Initial vocabulary with word frequencies and space-separated characters
vocab = {'h u g </w>': 5, 'p u g </w>': 2, 'p u n </w>': 3}

num_merges = 3
merges = []

for i in range(num_merges):
    print(f'Iteration {i+1}')
    for word, freq in vocab.items():
        print(f'  old vocabulary word: {word}, Frequency: {freq}')
    pairs = get_stats(vocab)
    if not pairs:
        break
    # Find the most frequent pair
    best = max(pairs, key=pairs.get)
    print(f'Most frequent pair: {best}')
    vocab = merge_vocab(best, vocab)
    merges.append(best)
    print("New vocab, after merging into old vocabulary:")
    for word, freq in vocab.items():
        print(f'  new vocabulary word: {word}, Frequency: {freq}')

print('\nLearned Merges:', merges)



