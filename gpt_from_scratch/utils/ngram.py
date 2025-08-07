# utils/ngram.py
# ==============================
# N-gram language models (unigram to 4-gram).
# Supports Laplace smoothing, interpolation, and token generation.
# ==============================

import math
import random
from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n=2):
        """
        Initialize the N-gram model.

        Args:
            n (int): The maximum N-gram order (1 = unigram, 2 = bigram, etc.)
        """
        self.n = n
        self.ngram_counts = [defaultdict(int) for _ in range(n)]
        self.context_counts = [defaultdict(int) for _ in range(n)]
        self.vocab = set()
        print(f"NGram Model initialized with n = {n}")

    def train(self, token_sequences):
        """
        Train the model by counting n-gram frequencies.

        Args:
            token_sequences (List[List[str]]): Tokenized text (BPE tokens per word)
        """
        for seq in token_sequences:
            tokens = ["<s>"] * (self.n - 1) + [tok for tok in seq] + ["</s>"]
            self.vocab.update(tokens)

            for i in range(len(tokens)):
                for k in range(1, self.n + 1):
                    if i - k + 1 < 0:
                        continue
                    ngram = tuple(tokens[i - k + 1:i + 1])
                    context = ngram[:-1]
                    self.ngram_counts[k - 1][ngram] += 1
                    self.context_counts[k - 1][context] += 1

    def laplace_prob(self, ngram, k=1):
        """
        Compute Laplace-smoothed probability of an n-gram.

        Args:
            ngram (Tuple[str]): The n-gram to evaluate.
            k (float): Smoothing constant (default=1 for add-one).

        Returns:
            float: Probability of the n-gram.
        """
        order = len(ngram)
        context = ngram[:-1]

        count_ngram = self.ngram_counts[order - 1][ngram]
        count_context = self.context_counts[order - 1][context]

        V = len(self.vocab)
        prob = (count_ngram + k) / (count_context + k * V)
        return prob

    def interpolated_prob(self, ngram, lambda1=0.1, lambda2=0.3, lambda3=0.6):
        """
        Compute interpolated probability combining uni-, bi-, tri-gram.

        Args:
            ngram (Tuple[str]): The trigram context.
            lambda1, lambda2, lambda3: Weights for unigram, bigram, trigram

        Returns:
            float: Interpolated probability
        """
        trigram = ngram
        bigram = ngram[1:]
        unigram = (ngram[2],)

        p1 = self.laplace_prob(unigram)
        p2 = self.laplace_prob(bigram)
        p3 = self.laplace_prob(trigram)

        return lambda1 * p1 + lambda2 * p2 + lambda3 * p3

    def perplexity(self, token_sequences, lambda1=0.1, lambda2=0.3, lambda3=0.6):
        """
        Compute perplexity over a list of tokenized sequences.

        Returns:
            float: Perplexity score (lower is better)
        """
        log_prob_sum = 0
        token_count = 0

        for seq in token_sequences:
            tokens = ["<s>"] * (self.n - 1) + [tok for tok in seq] + ["</s>"]
            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - 2:i + 1])
                prob = self.interpolated_prob(context, lambda1, lambda2, lambda3)
                log_prob_sum += -math.log(prob)
                token_count += 1

        return math.exp(log_prob_sum / token_count)

    def generate(self, context=None, max_tokens=10, method="argmax"):
        """
        Generate a sequence of tokens starting from a given context.

        Args:
            context (List[str]): Starting tokens
            max_tokens (int): Maximum length of output
            method (str): 'argmax' or 'sample'

        Returns:
            str: Generated token sequence
        """
        if context is None:
            context = ["<s>"] * (self.n - 1)
        else:
            context = ["<s>"] * max(0, self.n - 1 - len(context)) + context

        output = []

        for _ in range(max_tokens):
            candidates = {}
            for token in self.vocab:
                ngram = tuple(context[-(self.n - 1):] + [token])
                prob = self.laplace_prob(ngram)
                candidates[token] = prob

            if not candidates:
                break

            if method == "argmax":
                next_token = max(candidates, key=candidates.get)
            elif method == "sample":
                tokens, probs = zip(*candidates.items())
                total = sum(probs)
                probs = [p / total for p in probs]
                next_token = random.choices(tokens, weights=probs, k=1)[0]
            else:
                raise ValueError("Unknown generation method.")

            if next_token == "</s>":
                break

            output.append(next_token)
            context.append(next_token)

        return output
