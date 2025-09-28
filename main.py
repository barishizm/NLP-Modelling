"""Comprehensive n-gram language modelling lab implementation.

This script follows the step-by-step lab brief:

Part A  - download and preprocess a corpus, report the top tokens.
Part B  - build unigram/bigram/trigram models and score a test sentence.
Part C  - illustrate zero probabilities and apply Laplace smoothing.
Part D  - implement Katz-style back-off and linear interpolation models.
Part E  - evaluate the trigram model on a held-out test set via perplexity.

Running ``python main.py`` will print the artefacts requested in the lab.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from urllib.request import urlopen


CORPUS_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
TEST_SENTENCE = "students love learning natural language processing"


def download_corpus(url: str = CORPUS_URL) -> str:
    """Download a text corpus and return it as a Unicode string."""

    with urlopen(url) as response:
        return response.read().decode("utf-8", errors="replace")


def preprocess(text: str) -> Tuple[List[str], str]:
    """Lowercase, remove punctuation/digits, and tokenize the corpus.

    Returns the cleaned token list and a brief textual explanation of the
    preprocessing pipeline.
    """

    lowered = text.lower()
    # Keep letters and whitespace only; replace everything else with a space.
    cleaned = re.sub(r"[^a-z\s]", " ", lowered)
    tokens = cleaned.split()

    explanation = (
        "Lowercased the corpus, stripped punctuation and digits with a regular "
        "expression (keeping alphabetic characters and whitespace), and "
        "tokenised on whitespace."
    )

    return tokens, explanation


def ngram_counts(tokens: Sequence[str], n: int) -> Counter[Tuple[str, ...]]:
    """Compute n-gram counts for the given order ``n``."""

    counts: Counter[Tuple[str, ...]] = Counter()
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i : i + n])] += 1
    return counts


def unigram_probabilities(unigram_counts: Counter[Tuple[str, ...]]) -> Dict[str, float]:
    total = sum(unigram_counts.values())
    return {word: count / total for (word,), count in unigram_counts.items()}


def bigram_probabilities(
    bigram_counts: Counter[Tuple[str, str]],
    unigram_counts: Counter[Tuple[str, ...]],
) -> Dict[Tuple[str, str], float]:
    probs: Dict[Tuple[str, str], float] = {}
    for (w1, w2), count in bigram_counts.items():
        denom = unigram_counts.get((w1,), 0)
        if denom:
            probs[(w1, w2)] = count / denom
    return probs


def trigram_probabilities(
    trigram_counts: Counter[Tuple[str, str, str]],
    bigram_counts: Counter[Tuple[str, str]],
) -> Dict[Tuple[str, str, str], float]:
    probs: Dict[Tuple[str, str, str], float] = {}
    for (w1, w2, w3), count in trigram_counts.items():
        denom = bigram_counts.get((w1, w2), 0)
        if denom:
            probs[(w1, w2, w3)] = count / denom
    return probs


def sentence_probability_unigram(
    tokens: Sequence[str], uni_probs: Dict[str, float]
) -> Tuple[float, float, List[Tuple[str, float]]]:
    steps: List[Tuple[str, float]] = []
    prob = 1.0
    log_prob = 0.0
    for word in tokens:
        p = uni_probs.get(word, 0.0)
        steps.append((word, p))
        if p == 0.0:
            return 0.0, float("-inf"), steps
        prob *= p
        log_prob += math.log(p)
    return prob, log_prob, steps


def sentence_probability_bigram(
    tokens: Sequence[str],
    bi_probs: Dict[Tuple[str, str], float],
    unigram_counts: Counter[Tuple[str, ...]],
) -> Tuple[float, float, List[Tuple[Tuple[str, str], float]]]:
    steps: List[Tuple[Tuple[str, str], float]] = []
    if len(tokens) < 2:
        return 1.0, 0.0, steps
    prob = 1.0
    log_prob = 0.0
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        p = bi_probs.get((w1, w2), 0.0)
        steps.append(((w1, w2), p))
        if p == 0.0:
            return 0.0, float("-inf"), steps
        prob *= p
        log_prob += math.log(p)
    return prob, log_prob, steps


def sentence_probability_trigram(
    tokens: Sequence[str],
    tri_probs: Dict[Tuple[str, str, str], float],
) -> Tuple[float, float, List[Tuple[Tuple[str, str, str], float]]]:
    steps: List[Tuple[Tuple[str, str, str], float]] = []
    if len(tokens) < 3:
        return 1.0, 0.0, steps
    prob = 1.0
    log_prob = 0.0
    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
        p = tri_probs.get((w1, w2, w3), 0.0)
        steps.append(((w1, w2, w3), p))
        if p == 0.0:
            return 0.0, float("-inf"), steps
        prob *= p
        log_prob += math.log(p)
    return prob, log_prob, steps


def build_vocab(tokens: Iterable[str], min_freq: int = 2) -> set[str]:
    counts = Counter(tokens)
    vocab = {token for token, freq in counts.items() if freq >= min_freq}
    vocab.add("<unk>")
    return vocab


def map_to_vocab(tokens: Iterable[str], vocab: set[str]) -> List[str]:
    return [token if token in vocab else "<unk>" for token in tokens]


def laplace_unigram_prob(
    token: str,
    unigram_counts: Counter[Tuple[str, ...]],
    total_tokens: int,
    vocab_size: int,
) -> float:
    return (unigram_counts.get((token,), 0) + 1) / (total_tokens + vocab_size)


def laplace_bigram_prob(
    w1: str,
    w2: str,
    bigram_counts: Counter[Tuple[str, str]],
    unigram_counts: Counter[Tuple[str, ...]],
    vocab_size: int,
) -> float:
    return (bigram_counts.get((w1, w2), 0) + 1) / (unigram_counts.get((w1,), 0) + vocab_size)


def laplace_trigram_prob(
    w1: str,
    w2: str,
    w3: str,
    trigram_counts: Counter[Tuple[str, str, str]],
    bigram_counts: Counter[Tuple[str, str]],
    vocab_size: int,
) -> float:
    return (trigram_counts.get((w1, w2, w3), 0) + 1) / (bigram_counts.get((w1, w2), 0) + vocab_size)


def find_zero_probability_examples(
    sentence_tokens: Sequence[str],
    bigram_counts: Counter[Tuple[str, str]],
    trigram_counts: Counter[Tuple[str, str, str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    bigram_examples: List[Tuple[str, str]] = []
    trigram_examples: List[Tuple[str, str, str]] = []

    for i in range(len(sentence_tokens) - 1):
        pair = (sentence_tokens[i], sentence_tokens[i + 1])
        if bigram_counts.get(pair, 0) == 0 and pair not in bigram_examples:
            bigram_examples.append(pair)
        if len(bigram_examples) >= 2:
            break

    for i in range(len(sentence_tokens) - 2):
        triple = (sentence_tokens[i], sentence_tokens[i + 1], sentence_tokens[i + 2])
        if trigram_counts.get(triple, 0) == 0 and triple not in trigram_examples:
            trigram_examples.append(triple)
        if len(trigram_examples) >= 2:
            break

    return bigram_examples, trigram_examples


def normalise_tokens(tokens: Iterable[str], vocab: set[str]) -> List[str]:
    return [token if token in vocab else "<unk>" for token in tokens]


def backoff_sentence_logprob(
    sentence_tokens: Sequence[str],
    vocab: set[str],
    unigram_counts: Counter[Tuple[str, ...]],
    bigram_counts: Counter[Tuple[str, str]],
    trigram_counts: Counter[Tuple[str, str, str]],
    total_tokens: int,
    vocab_size: int,
) -> Tuple[float, List[Tuple[Tuple[str, str, str], float, str]]]:
    steps: List[Tuple[Tuple[str, str, str], float, str]] = []
    sentence = normalise_tokens(sentence_tokens, vocab)
    log_prob = 0.0

    for i in range(len(sentence) - 2):
        w1, w2, w3 = sentence[i], sentence[i + 1], sentence[i + 2]
        if trigram_counts.get((w1, w2, w3), 0) > 0:
            p = laplace_trigram_prob(w1, w2, w3, trigram_counts, bigram_counts, vocab_size)
            source = "trigram"
        elif bigram_counts.get((w2, w3), 0) > 0:
            p = laplace_bigram_prob(w2, w3, bigram_counts, unigram_counts, vocab_size)
            source = "bigram"
        else:
            p = laplace_unigram_prob(w3, unigram_counts, total_tokens, vocab_size)
            source = "unigram"
        log_prob += math.log(p)
        steps.append(((w1, w2, w3), p, source))

    return log_prob, steps


def interpolation_sentence_logprob(
    sentence_tokens: Sequence[str],
    vocab: set[str],
    unigram_counts: Counter[Tuple[str, ...]],
    bigram_counts: Counter[Tuple[str, str]],
    trigram_counts: Counter[Tuple[str, str, str]],
    total_tokens: int,
    vocab_size: int,
    lambdas: Tuple[float, float, float] = (0.6, 0.3, 0.1),
) -> Tuple[float, List[Tuple[Tuple[str, str, str], float, Tuple[float, float, float]]]]:
    sentence = normalise_tokens(sentence_tokens, vocab)
    steps: List[Tuple[Tuple[str, str, str], float, Tuple[float, float, float]]] = []
    log_prob = 0.0

    for i in range(len(sentence) - 2):
        w1, w2, w3 = sentence[i], sentence[i + 1], sentence[i + 2]
        p_tri = laplace_trigram_prob(w1, w2, w3, trigram_counts, bigram_counts, vocab_size)
        p_bi = laplace_bigram_prob(w2, w3, bigram_counts, unigram_counts, vocab_size)
        p_uni = laplace_unigram_prob(w3, unigram_counts, total_tokens, vocab_size)
        p = lambdas[0] * p_tri + lambdas[1] * p_bi + lambdas[2] * p_uni
        log_prob += math.log(p)
        steps.append(((w1, w2, w3), p, (p_tri, p_bi, p_uni)))

    return log_prob, steps


def trigram_stream_logprob(
    tokens: Sequence[str],
    prob_fn,
) -> Tuple[float, int]:
    log_prob = 0.0
    count = 0
    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
        p = prob_fn(w1, w2, w3)
        if p <= 0.0:
            return float("-inf"), 0
        log_prob += math.log(p)
        count += 1
    return log_prob, count


def perplexity(tokens: Sequence[str], prob_fn) -> float:
    log_prob, n_events = trigram_stream_logprob(tokens, prob_fn)
    if n_events == 0 or not math.isfinite(log_prob):
        return float("inf")
    return math.exp(-log_prob / n_events)


@dataclass
class PartAResult:
    tokens: List[str]
    explanation: str
    top_tokens: List[Tuple[str, int]]


def part_a(text: str) -> PartAResult:
    tokens, explanation = preprocess(text)
    top_tokens = Counter(tokens).most_common(20)
    return PartAResult(tokens, explanation, top_tokens)


@dataclass
class PartBResult:
    unigram_steps: List[Tuple[str, float]]
    bigram_steps: List[Tuple[Tuple[str, str], float]]
    trigram_steps: List[Tuple[Tuple[str, str, str], float]]
    unigram_prob: float
    bigram_prob: float
    trigram_prob: float


def part_b(tokens: Sequence[str], sentence: str) -> PartBResult:
    sentence_tokens = preprocess(sentence)[0]
    uni_counts = ngram_counts(tokens, 1)
    bi_counts = ngram_counts(tokens, 2)
    tri_counts = ngram_counts(tokens, 3)

    uni_probs = unigram_probabilities(uni_counts)
    bi_probs = bigram_probabilities(bi_counts, uni_counts)
    tri_probs = trigram_probabilities(tri_counts, bi_counts)

    u_prob, _, u_steps = sentence_probability_unigram(sentence_tokens, uni_probs)
    b_prob, _, b_steps = sentence_probability_bigram(sentence_tokens, bi_probs, uni_counts)
    t_prob, _, t_steps = sentence_probability_trigram(sentence_tokens, tri_probs)

    return PartBResult(u_steps, b_steps, t_steps, u_prob, b_prob, t_prob)


@dataclass
class PartCResult:
    bigram_examples: List[Tuple[str, str, float, float]]
    trigram_examples: List[Tuple[str, str, str, float, float]]
    vocab: set[str]
    unigram_counts: Counter[Tuple[str, ...]]
    bigram_counts: Counter[Tuple[str, str]]
    trigram_counts: Counter[Tuple[str, str, str]]
    total_tokens: int


def part_c(tokens: Sequence[str], sentence: str) -> PartCResult:
    sentence_tokens = preprocess(sentence)[0]
    uni_counts = ngram_counts(tokens, 1)
    bi_counts = ngram_counts(tokens, 2)
    tri_counts = ngram_counts(tokens, 3)

    bigram_zero, trigram_zero = find_zero_probability_examples(
        sentence_tokens, bi_counts, tri_counts
    )

    vocab = build_vocab(tokens, min_freq=2)
    mapped_tokens = map_to_vocab(tokens, vocab)
    unigram_smoothed = ngram_counts(mapped_tokens, 1)
    bigram_smoothed = ngram_counts(mapped_tokens, 2)
    trigram_smoothed = ngram_counts(mapped_tokens, 3)
    total_tokens = sum(unigram_smoothed.values())
    vocab_size = len(vocab)

    bigram_examples: List[Tuple[str, str, float, float]] = []
    for w1, w2 in bigram_zero:
        before = 0.0
        after = laplace_bigram_prob(
            w1 if w1 in vocab else "<unk>",
            w2 if w2 in vocab else "<unk>",
            bigram_smoothed,
            unigram_smoothed,
            vocab_size,
        )
        bigram_examples.append((w1, w2, before, after))

    trigram_examples: List[Tuple[str, str, str, float, float]] = []
    for w1, w2, w3 in trigram_zero:
        before = 0.0
        after = laplace_trigram_prob(
            w1 if w1 in vocab else "<unk>",
            w2 if w2 in vocab else "<unk>",
            w3 if w3 in vocab else "<unk>",
            trigram_smoothed,
            bigram_smoothed,
            vocab_size,
        )
        trigram_examples.append((w1, w2, w3, before, after))

    # Ensure at least two examples by adding synthetic ones if necessary.
    extra_bigrams = [("alice", "students"), ("quantum", "cat")]
    for w1, w2 in extra_bigrams:
        if len(bigram_examples) >= 2:
            break
        after = laplace_bigram_prob(
            w1 if w1 in vocab else "<unk>",
            w2 if w2 in vocab else "<unk>",
            bigram_smoothed,
            unigram_smoothed,
            vocab_size,
        )
        bigram_examples.append((w1, w2, 0.0, after))

    extra_trigrams = [("tea", "party", "students"), ("white", "rabbit", "nlp")]
    for w1, w2, w3 in extra_trigrams:
        if len(trigram_examples) >= 2:
            break
        after = laplace_trigram_prob(
            w1 if w1 in vocab else "<unk>",
            w2 if w2 in vocab else "<unk>",
            w3 if w3 in vocab else "<unk>",
            trigram_smoothed,
            bigram_smoothed,
            vocab_size,
        )
        trigram_examples.append((w1, w2, w3, 0.0, after))

    return PartCResult(
        bigram_examples,
        trigram_examples,
        vocab,
        unigram_smoothed,
        bigram_smoothed,
        trigram_smoothed,
        total_tokens,
    )


@dataclass
class PartDResult:
    sentences: List[str]
    backoff: List[Tuple[float, List[Tuple[Tuple[str, str, str], float, str]]]]
    interpolation: List[
        Tuple[float, List[Tuple[Tuple[str, str, str], float, Tuple[float, float, float]]]]
    ]


def part_d(part_c: PartCResult, sentences: Sequence[str]) -> PartDResult:
    vocab = part_c.vocab
    unigram_counts = part_c.unigram_counts
    bigram_counts = part_c.bigram_counts
    trigram_counts = part_c.trigram_counts
    total_tokens = part_c.total_tokens
    vocab_size = len(vocab)

    backoff_results: List[Tuple[float, List[Tuple[Tuple[str, str, str], float, str]]]] = []
    interpolation_results: List[
        Tuple[float, List[Tuple[Tuple[str, str, str], float, Tuple[float, float, float]]]]
    ] = []

    for sentence in sentences:
        sent_tokens = preprocess(sentence)[0]
        bo_log_prob, bo_steps = backoff_sentence_logprob(
            sent_tokens,
            vocab,
            unigram_counts,
            bigram_counts,
            trigram_counts,
            total_tokens,
            vocab_size,
        )
        it_log_prob, it_steps = interpolation_sentence_logprob(
            sent_tokens,
            vocab,
            unigram_counts,
            bigram_counts,
            trigram_counts,
            total_tokens,
            vocab_size,
        )
        backoff_results.append((bo_log_prob, bo_steps))
        interpolation_results.append((it_log_prob, it_steps))

    return PartDResult(list(sentences), backoff_results, interpolation_results)


@dataclass
class PartEResult:
    perplexity_mle: float
    perplexity_laplace: float
    perplexity_interpolation: float


def part_e(tokens: Sequence[str], part_c: PartCResult) -> PartEResult:
    split_idx = int(0.8 * len(tokens))
    train_raw = tokens[:split_idx]
    test_raw = tokens[split_idx:]

    vocab = build_vocab(train_raw, min_freq=2)
    train = map_to_vocab(train_raw, vocab)
    test = map_to_vocab(test_raw, vocab)

    unigram_counts = ngram_counts(train, 1)
    bigram_counts = ngram_counts(train, 2)
    trigram_counts = ngram_counts(train, 3)
    total_tokens = sum(unigram_counts.values())
    vocab_size = len(vocab)

    def trigram_mle_prob(w1: str, w2: str, w3: str) -> float:
        num = trigram_counts.get((w1, w2, w3), 0)
        denom = bigram_counts.get((w1, w2), 0)
        return (num / denom) if denom and num else 0.0

    def trigram_laplace_prob(w1: str, w2: str, w3: str) -> float:
        return laplace_trigram_prob(w1, w2, w3, trigram_counts, bigram_counts, vocab_size)

    def trigram_interpolation_prob(w1: str, w2: str, w3: str) -> float:
        p_tri = trigram_laplace_prob(w1, w2, w3)
        p_bi = laplace_bigram_prob(w2, w3, bigram_counts, unigram_counts, vocab_size)
        p_uni = laplace_unigram_prob(w3, unigram_counts, total_tokens, vocab_size)
        return 0.6 * p_tri + 0.3 * p_bi + 0.1 * p_uni

    perp_mle = perplexity(test, trigram_mle_prob)
    perp_laplace = perplexity(test, trigram_laplace_prob)
    perp_interp = perplexity(test, trigram_interpolation_prob)

    return PartEResult(perp_mle, perp_laplace, perp_interp)


def main() -> None:
    print("Downloading corpus...")
    raw_text = download_corpus()

    print("\n=== PART A: Preprocessing ===")
    part_a_result = part_a(raw_text)
    print(part_a_result.explanation)
    print(f"Total tokens after cleaning: {len(part_a_result.tokens):,}")
    print("Top 20 tokens:")
    for token, count in part_a_result.top_tokens:
        print(f"  {token:>15s} : {count}")
    print("Sample of cleaned tokens:")
    print(part_a_result.tokens[:40])

    print("\n=== PART B: Sentence Probabilities ===")
    part_b_result = part_b(part_a_result.tokens, TEST_SENTENCE)
    print(f"Sentence: '{TEST_SENTENCE}'")
    print("Unigram probability steps:")
    for token, prob in part_b_result.unigram_steps:
        print(f"  P({token}) = {prob:.6g}")
    print(f"Total unigram probability: {part_b_result.unigram_prob:.6e}")

    print("Bigram probability steps:")
    for (w1, w2), prob in part_b_result.bigram_steps:
        print(f"  P({w2} | {w1}) = {prob:.6g}")
    print(f"Total bigram probability: {part_b_result.bigram_prob:.6e}")

    print("Trigram probability steps:")
    for (w1, w2, w3), prob in part_b_result.trigram_steps:
        print(f"  P({w3} | {w1}, {w2}) = {prob:.6g}")
    print(f"Total trigram probability: {part_b_result.trigram_prob:.6e}")

    print("\n=== PART C: Handling Sparsity with Laplace Smoothing ===")
    part_c_result = part_c(part_a_result.tokens, TEST_SENTENCE)
    print("Zero-probability bigrams before smoothing and after Laplace:")
    for w1, w2, before, after in part_c_result.bigram_examples[:2]:
        print(f"  ({w1}, {w2}): before={before:.1f}, after={after:.6g}")
    print("Zero-probability trigrams before smoothing and after Laplace:")
    for w1, w2, w3, before, after in part_c_result.trigram_examples[:2]:
        print(f"  ({w1}, {w2}, {w3}): before={before:.1f}, after={after:.6g}")

    print("\n=== PART D: Back-off vs. Interpolation ===")
    sentences = [
        TEST_SENTENCE,
        "alice was beginning to get very tired",
        "the white rabbit was late",
    ]
    part_d_result = part_d(part_c_result, sentences)
    for sent, (bo_log, bo_steps), (it_log, it_steps) in zip(
        part_d_result.sentences,
        part_d_result.backoff,
        part_d_result.interpolation,
    ):
        print(f"Sentence: '{sent}'")
        print(f"  Back-off log-probability: {bo_log:.6f}")
        for context, prob, source in bo_steps:
            print(f"    {context} -> {source:>8s}, p={prob:.6g}")
        print(f"  Interpolation log-probability: {it_log:.6f}")
        for context, prob, components in it_steps:
            tri_p, bi_p, uni_p = components
            print(
                "    {} -> mix, p={:.6g} (tri={:.6g}, bi={:.6g}, uni={:.6g})".format(
                    context, prob, tri_p, bi_p, uni_p
                )
            )

    print("\n=== PART E: Trigram Perplexity on Held-out Data ===")
    part_e_result = part_e(part_a_result.tokens, part_c_result)
    print(f"Perplexity (MLE trigram): {part_e_result.perplexity_mle}")
    print(f"Perplexity (Laplace trigram): {part_e_result.perplexity_laplace:.3f}")
    print(f"Perplexity (Interpolated trigram): {part_e_result.perplexity_interpolation:.3f}")


if __name__ == "__main__":
    main()

