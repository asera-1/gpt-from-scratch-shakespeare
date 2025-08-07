# utils/bpe.py

import re
from collections import defaultdict
import unicodedata

class BPETokenizer:
    def __init__(self, num_merges=500):
        self.num_merges = num_merges
        self.merges = []
        self.token2id = {}
        self.id2token = {}

    def normalize(self, text):
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = text.lower()
        text = re.sub(r'\d+', '<num>', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_vocab(self, text):
        vocab = defaultdict(int)
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            token = " ".join(list(word)) + " </w>"
            vocab[token] += 1
        return vocab

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        pattern = re.escape(' '.join(pair))
        replacement = ''.join(pair)
        new_vocab = {}
        for word in vocab:
            new_word = re.sub(r'(?<!\S)' + pattern + r'(?!\S)', replacement, word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def learn_bpe(self, text):
        vocab = self.get_vocab(text)
        for _ in range(self.num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            vocab = self.merge_vocab(best_pair, vocab)

    def segment_word(self, word):
        symbols = list(word) + ['</w>']
        while True:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            merge_found = False
            for merge in self.merges:
                if merge in pairs:
                    i = pairs.index(merge)
                    symbols = symbols[:i] + [''.join(merge)] + symbols[i+2:]
                    merge_found = True
                    break
            if not merge_found:
                break
        return symbols

    def segment_text(self, text):
        words = re.findall(r'\b\w+\b', text)
        return [self.segment_word(word) for word in words]

    def encode(self, text):
        tokens = self.segment_text(text)
        flat_tokens = [tok for word in tokens for tok in word]
        # Map to token IDs
        ids = []
        for tok in flat_tokens:
            if tok not in self.token2id:
                self.token2id[tok] = len(self.token2id)
                self.id2token[self.token2id[tok]] = tok
            ids.append(self.token2id[tok])
        return ids

    def decode(self, ids):
        tokens = [self.id2token[i] for i in ids]
        text = ' '.join(tokens).replace('</w>', '').replace(' ', '')
        return text
