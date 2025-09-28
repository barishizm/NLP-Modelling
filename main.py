# Full N-gram NLP Lab Pipeline — English Only
# Parts A–E: Preprocessing, N-gram MLE, Smoothing, Back-off & Interpolation, Perplexity, and Plots

import os
import re
import math
from pathlib import Path
from typing import Optional, Union
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib.pyplot as plt
from collections import Counter

# -------------------------------------
# Common: Stopwords (simple, illustrative)
# -------------------------------------
stop_words = {
    "the","and","to","of","a","in","that","it","is","was","he","for","on",
    "are","as","with","his","they","i","at","be","this","have","from","or",
    "one","had","by","word","but","not","s","t"
}

# =====================================
# PART A — Download, preprocess, top-20
# =====================================
DEFAULT_CORPUS_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
DEFAULT_FALLBACK_PATH = Path("data/alice.txt")


def download_corpus(
    url: str = DEFAULT_CORPUS_URL,
    local_override: Optional[Union[Path, str]] = None,
) -> str:
    """Return the raw corpus text.

    When HTTPS is unavailable you can either pass a ``Path`` pointing to a local
    copy of the corpus, or set the ``ALICE_CORPUS_PATH`` environment variable to
    the desired file.  As a final fallback the function will try reading from
    ``data/alice.txt`` if it exists.  If all options fail a descriptive error is
    raised so lab runners know how to proceed.
    """

    # Allow users to provide a local file explicitly via argument or env var
    override = local_override or os.getenv("ALICE_CORPUS_PATH")
    if override:
        path = Path(override)
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(
            f"Local corpus override '{path}' was not found. Provide a valid path "
            "or remove the override to download the default corpus."
        )

    try:
        with urlopen(url, timeout=60) as response:
            return response.read().decode("utf-8")
    except URLError as exc:
        # HTTPS may be blocked (e.g. in offline grading environments).  Try a
        # cached copy before surfacing a friendly error.
        if DEFAULT_FALLBACK_PATH.exists():
            return DEFAULT_FALLBACK_PATH.read_text(encoding="utf-8")
        raise RuntimeError(
            "Failed to download the corpus. Provide a local file via the "
            "'local_override' argument or set the ALICE_CORPUS_PATH environment "
            "variable pointing to a cached copy."
        ) from exc


text = download_corpus()

# Lowercase
text = text.lower()

# Remove punctuation and digits (keep letters and spaces)
text = re.sub(r"[^a-z\s]", " ", text)

# Tokenize
tokens = text.split()

# Stopword removal (optional for Part A)
tokens = [w for w in tokens if w not in stop_words]

# Top-20 frequent tokens
freq = Counter(tokens)
top20 = freq.most_common(20)

print("Top 20 tokens (stopwords removed):")
for word, count in top20:
    print(f"{word}: {count}")

# -------------------------------------
# Helper: reuse already-built tokens or rebuild quickly
# -------------------------------------
def prepare_tokens_if_needed():
    try:
        tokens  # noqa: F821
        assert isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], str)
        return tokens
    except Exception:
        text = download_corpus()
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters & spaces
        toks = text.split()
        return toks

tokens = prepare_tokens_if_needed()

# =====================================
# PART B — N-gram counts and MLE sentence probabilities
# (No start/end tokens, as per the lab note)
# =====================================
def ngram_counts(toks, n):
    c = Counter()
    for i in range(len(toks) - n + 1):
        c[tuple(toks[i:i+n])] += 1
    return c

uni_counts = ngram_counts(tokens, 1)
bi_counts  = ngram_counts(tokens, 2)
tri_counts = ngram_counts(tokens, 3)

N = sum(uni_counts.values())  # total tokens

# Unigram MLE
uni_prob = { (w,): cnt / N for (w,), cnt in uni_counts.items() }

# Bigram MLE
bi_prob = {}
for (w1, w2), num in bi_counts.items():
    denom = uni_counts.get((w1,), 0)
    if denom > 0:
        bi_prob[(w1, w2)] = num / denom

# Trigram MLE
tri_prob = {}
for (w1, w2, w3), num in tri_counts.items():
    denom = bi_counts.get((w1, w2), 0)
    if denom > 0:
        tri_prob[(w1, w2, w3)] = num / denom

def clean_sentence(s):
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return s.split()

def sent_prob_unigram(sent_toks):
    p = 1.0
    logp = 0.0
    steps = []
    for w in sent_toks:
        pr = uni_prob.get((w,), 0.0)
        steps.append((w, pr))
        if pr == 0.0:
            return 0.0, float("-inf"), steps
        p *= pr
        logp += math.log(pr)
    return p, logp, steps

def sent_prob_bigram(sent_toks):
    if len(sent_toks) < 2:
        return 1.0, 0.0, []
    p = 1.0
    logp = 0.0
    steps = []
    for i in range(len(sent_toks) - 1):
        w1, w2 = sent_toks[i], sent_toks[i+1]
        num = bi_counts.get((w1, w2), 0)
        denom = uni_counts.get((w1,), 0)
        pr = (num / denom) if denom > 0 and num > 0 else 0.0
        steps.append(((w1, w2), pr))
        if pr == 0.0:
            return 0.0, float("-inf"), steps
        p *= pr
        logp += math.log(pr)
    return p, logp, steps

def sent_prob_trigram(sent_toks):
    if len(sent_toks) < 3:
        return 1.0, 0.0, []
    p = 1.0
    logp = 0.0
    steps = []
    for i in range(len(sent_toks) - 2):
        w1, w2, w3 = sent_toks[i], sent_toks[i+1], sent_toks[i+2]
        num = tri_counts.get((w1, w2, w3), 0)
        denom = bi_counts.get((w1, w2), 0)
        pr = (num / denom) if denom > 0 and num > 0 else 0.0
        steps.append(((w1, w2, w3), pr))
        if pr == 0.0:
            return 0.0, float("-inf"), steps
        p *= pr
        logp += math.log(pr)
    return p, logp, steps

# Test sentence for Part B
test_sentence = "students love learning natural language processing"
stoks = clean_sentence(test_sentence)

u_p, u_logp, u_steps = sent_prob_unigram(stoks)
b_p, b_logp, b_steps = sent_prob_bigram(stoks)
t_p, t_logp, t_steps = sent_prob_trigram(stoks)

print("Sentence tokens:", stoks, "\n")
print("=== UNIGRAM ===")
for w, pr in u_steps:
    print(f"P({w}) = {pr:.6g}")
print(f"Sentence P_unigram = {u_p:.6e} ; log = {u_logp:.6f}\n")

print("=== BIGRAM ===")
for (w1, w2), pr in b_steps:
    print(f"P({w2} | {w1}) = {pr:.6g}")
print(f"Sentence P_bigram = {b_p:.6e} ; log = {b_logp:.6f}\n")

print("=== TRIGRAM ===")
for (w1, w2, w3), pr in t_steps:
    print(f"P({w3} | {w1}, {w2}) = {pr:.6g}")
print(f"Sentence P_trigram = {t_p:.6e} ; log = {t_logp:.6f}\n")

# =======================================================
# PART C — Sparsity & Smoothing (UNK mapping + Add-1)
# =======================================================
def prepare_tokens_if_needed():
    try:
        tokens  # noqa
        assert isinstance(tokens, list) and len(tokens) > 0
        return tokens
    except Exception:
        url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
        text = requests.get(url, timeout=30).text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.split()

tokens = prepare_tokens_if_needed()

def ngram_counts(toks, n):
    c = Counter()
    for i in range(len(toks)-n+1):
        c[tuple(toks[i:i+n])] += 1
    return c

# Raw counts without UNK (MLE helpers)
uni_c_raw = ngram_counts(tokens, 1)
bi_c_raw  = ngram_counts(tokens, 2)
tri_c_raw = ngram_counts(tokens, 3)

def mle_bigram_prob(w1, w2):
    num = bi_c_raw.get((w1, w2), 0)
    den = uni_c_raw.get((w1,), 0)
    return (num/den) if den > 0 and num > 0 else 0.0

def mle_trigram_prob(w1, w2, w3):
    num = tri_c_raw.get((w1, w2, w3), 0)
    den = bi_c_raw.get((w1, w2), 0)
    return (num/den) if den > 0 and num > 0 else 0.0

# Build vocabulary with UNK
def build_vocab(toks, min_freq=2):
    cnt = Counter(toks)
    vocab = {w for w, f in cnt.items() if f >= min_freq}
    vocab.add("<unk>")
    return vocab

def map_to_unk(toks, vocab):
    return [w if w in vocab else "<unk>" for w in toks]

vocab = build_vocab(tokens, min_freq=2)
tok_unk = map_to_unk(tokens, vocab)
V = len(vocab)

uni_c = ngram_counts(tok_unk, 1)
bi_c  = ngram_counts(tok_unk, 2)
tri_c = ngram_counts(tok_unk, 3)
N = sum(uni_c.values())

# Laplace (add-1) smoothed probabilities
def laplace_uni(w):
    return (uni_c.get((w,), 0) + 1) / (N + V)

def laplace_bi(w1, w2):
    return (bi_c.get((w1, w2), 0) + 1) / (uni_c.get((w1,), 0) + V)

def laplace_tri(w1, w2, w3):
    return (tri_c.get((w1, w2, w3), 0) + 1) / (bi_c.get((w1, w2), 0) + V)

def norm(w):  # map OOV test tokens to <unk>
    return w if w in vocab else "<unk>"

# Choose n-grams for before/after reporting
test_sentence = "students love learning natural language processing"
stoks = re.sub(r"[^a-z\s]", " ", test_sentence.lower()).split()

bigram_tests = []
trigram_tests = []
# from the sentence
for i in range(len(stoks)-1):
    bigram_tests.append((stoks[i], stoks[i+1]))
for i in range(len(stoks)-2):
    trigram_tests.append((stoks[i], stoks[i+1], stoks[i+2]))
# extra (likely unseen)
bigram_tests += [("quantum", "cat"), ("alice", "students")]
trigram_tests += [("tea", "party", "students"), ("white", "rabbit", "nlp")]

print("Vocabulary size (with <unk>):", V)
print("Example OOV mapping for test tokens:",
      [f"{w}->{norm(w)}" for w in stoks], "\n")

print("=== BIGRAMS: BEFORE (MLE) vs AFTER (Laplace) ===")
shown_bi = 0
for w1, w2 in bigram_tests:
    p_before = mle_bigram_prob(w1, w2)
    p_after  = laplace_bi(norm(w1), norm(w2))
    if p_before == 0.0:  # report zero-probability examples
        pair = f"({w1}, {w2})"
        print(f"{pair:>30} | before={p_before:.0f}   after={p_after:.6g}")
        shown_bi += 1
    if shown_bi >= 2:
        break
if shown_bi == 0:
    print("Note: Did not find zero-probability bigrams (rare).")

print("\n=== TRIGRAMS: BEFORE (MLE) vs AFTER (Laplace) ===")
shown_tri = 0
for w1, w2, w3 in trigram_tests:
    p_before = mle_trigram_prob(w1, w2, w3)
    p_after  = laplace_tri(norm(w1), norm(w2), norm(w3))
    if p_before == 0.0:
        tri = f"({w1}, {w2}, {w3})"
        print(f"{tri:>40} | before={p_before:.0f}   after={p_after:.6g}")
        shown_tri += 1
    if shown_tri >= 2:
        break
if shown_tri == 0:
    print("Note: Did not find zero-probability trigrams (rare).")

# =======================================================
# PART D — Back-off and Linear Interpolation
# (using Laplace-smoothed probabilities with UNK mapping)
# =======================================================
def backoff_prob(sent_tokens):
    lp = 0.0
    steps = []
    for i in range(len(sent_tokens) - 2):
        w1, w2, w3 = map(norm, sent_tokens[i:i + 3])

        # If trigram count > 0, use trigram; else if bigram > 0, use bigram; else use unigram
        tri_p = tri_c.get((w1, w2, w3), 0)
        if tri_p > 0:
            p = laplace_tri(w1, w2, w3)
            src = "trigram"
        else:
            bi_p = bi_c.get((w2, w3), 0)
            if bi_p > 0:
                p = laplace_bi(w2, w3)
                src = "bigram"
            else:
                p = laplace_uni(w3)
                src = "unigram"
        lp += math.log(p)
        steps.append(((w1, w2, w3), p, src))
    return lp, steps

def interpolation_prob(sent_tokens, l1=0.6, l2=0.3, l3=0.1):
    lp = 0.0
    steps = []
    for i in range(len(sent_tokens) - 2):
        w1, w2, w3 = map(norm, sent_tokens[i:i + 3])
        p_tri = laplace_tri(w1, w2, w3)
        p_bi  = laplace_bi(w2, w3)
        p_uni = laplace_uni(w3)
        p = l1 * p_tri + l2 * p_bi + l3 * p_uni
        lp += math.log(p)
        steps.append(((w1, w2, w3), p, (p_tri, p_bi, p_uni)))
    return lp, steps

test_sentences = [
    "students love learning natural language processing",
    "alice was beginning to get very tired",
    "the white rabbit was late"
]

for s in test_sentences:
    stoks = re.sub(r"[^a-z\s]", " ", s.lower()).split()
    print("\nSentence:", s)

    bo_lp, bo_steps = backoff_prob(stoks)
    it_lp, it_steps = interpolation_prob(stoks)

    print(" Back-off logP:", bo_lp)
    for (w1, w2, w3), p, src in bo_steps:
        print(f"   {w1, w2, w3} -> {src}, p={p:.6g}")

    print(" Interpolation logP:", it_lp)
    for (w1, w2, w3), p, (pt, pb, pu) in it_steps:
        print(f"   {w1, w2, w3} -> mix, p={p:.6g} (tri={pt:.6g}, bi={pb:.6g}, uni={pu:.6g})")

# =====================================
# PART E — Perplexity (train/test split)
# =====================================
def prepare_tokens_if_needed():
    try:
        tokens  # noqa
        assert isinstance(tokens, list) and len(tokens) > 0
        return tokens
    except Exception:
        url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
        text = requests.get(url, timeout=30).text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.split()

tokens_all = prepare_tokens_if_needed()

# Train / test split
split = int(0.8 * len(tokens_all))
train_tokens_raw = tokens_all[:split]
test_tokens_raw  = tokens_all[split:]

# Vocabulary from train, then map train/test to UNK
def build_vocab(toks, min_freq=2):
    cnt = Counter(toks)
    vocab = {w for w, f in cnt.items() if f >= min_freq}
    vocab.add("<unk>")
    return vocab

def map_to_unk(toks, vocab):
    return [w if w in vocab else "<unk>" for w in toks]

vocab = build_vocab(train_tokens_raw, min_freq=2)
train_tokens = map_to_unk(train_tokens_raw, vocab)
test_tokens  = map_to_unk(test_tokens_raw,  vocab)
V = len(vocab)

# N-gram counts on train
def ngram_counts(toks, n):
    c = Counter()
    for i in range(len(toks)-n+1):
        c[tuple(toks[i:i+n])] += 1
    return c

uni_c = ngram_counts(train_tokens, 1)
bi_c  = ngram_counts(train_tokens, 2)
tri_c = ngram_counts(train_tokens, 3)
N = sum(uni_c.values())

# Smoothed and MLE trigram helpers
def laplace_uni(w):
    return (uni_c.get((w,), 0) + 1) / (N + V)

def laplace_bi(w1, w2):
    return (bi_c.get((w1, w2), 0) + 1) / (uni_c.get((w1,), 0) + V)

def laplace_tri(w1, w2, w3):
    return (tri_c.get((w1, w2, w3), 0) + 1) / (bi_c.get((w1, w2), 0) + V)

def mle_tri(w1, w2, w3):
    num = tri_c.get((w1, w2, w3), 0)
    den = bi_c.get((w1, w2), 0)
    return (num/den) if den > 0 and num > 0 else 0.0

def trigram_logprob_stream(toks, prob_fn):
    """Compute log-probability over a token stream for trigrams."""
    lp = 0.0
    count = 0
    for i in range(len(toks) - 2):
        w1, w2, w3 = toks[i], toks[i+1], toks[i+2]
        p = prob_fn(w1, w2, w3)
        if p <= 0.0:
            return float("-inf"), 0  # MLE zero -> -inf
        lp += math.log(p)
        count += 1
    return lp, count

def perplexity_trigram(toks, prob_fn):
    lp, m = trigram_logprob_stream(toks, prob_fn)
    if m == 0 or not math.isfinite(lp):
        return float("inf")
    avg_neg_log = - lp / m
    return math.exp(avg_neg_log)

def interpolation_prob_fn(l1=0.6, l2=0.3, l3=0.1):
    def f(w1, w2, w3):
        return l1*laplace_tri(w1, w2, w3) + l2*laplace_bi(w2, w3) + l3*laplace_uni(w3)
    return f

pp_mle    = perplexity_trigram(test_tokens, mle_tri)  # likely inf
pp_lap    = perplexity_trigram(test_tokens, laplace_tri)
pp_interp = perplexity_trigram(test_tokens, interpolation_prob_fn(0.6, 0.3, 0.1))

print("Vocab size (train):", V)
print("Test token count (effective trigrams):", max(0, len(test_tokens)-2))
print("\nPerplexity (test set):")
print("  Trigram MLE (unsmoothed):", pp_mle)
print("  Trigram + Laplace:", pp_lap)
print("  Interpolation (λ=0.6/0.3/0.1):", pp_interp)

# =======================================================
# PART F — Unified pipeline with plots for A–E
# (English-only version)
# =======================================================
def load_or_use_tokens():
    try:
        tokens  # noqa: F821
        assert isinstance(tokens, list) and tokens and isinstance(tokens[0], str)
        return tokens
    except Exception:
        url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
        text = requests.get(url, timeout=30).text
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.split()

tokens_all = load_or_use_tokens()

# Part A: Top-20 tokens (raw) — Plot
cnt_all = Counter(tokens_all)
top20 = cnt_all.most_common(20)

plt.figure()
plt.bar([w for w, _ in top20], [c for _, c in top20])
plt.xticks(rotation=70)
plt.title("Part A — Top 20 Tokens (raw)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# N-gram helpers shared by plots
def ngram_counts(toks, n):
    c = Counter()
    for i in range(len(toks)-n+1):
        c[tuple(toks[i:i+n])] += 1
    return c

def build_vocab(toks, min_freq=2):
    cnt = Counter(toks)
    vocab = {w for w, f in cnt.items() if f >= min_freq}
    vocab.add("<unk>")
    return vocab

def map_to_unk(toks, vocab):
    return [w if w in vocab else "<unk>" for w in toks]

def clean_sentence_to_tokens(s):
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return s.split()

# Part B (unsmoothed) — step probabilities for a sentence
uni_c_B = ngram_counts(tokens_all, 1)
bi_c_B  = ngram_counts(tokens_all, 2)
tri_c_B = ngram_counts(tokens_all, 3)
N_B = sum(uni_c_B.values())

def uni_p_mle_B(w):
    return uni_c_B.get((w,), 0) / N_B if N_B > 0 else 0.0

def bi_p_mle_B(w1, w2):
    num = bi_c_B.get((w1, w2), 0)
    den = uni_c_B.get((w1,), 0)
    return (num/den) if den > 0 and num > 0 else 0.0

def tri_p_mle_B(w1, w2, w3):
    num = tri_c_B.get((w1, w2, w3), 0)
    den = bi_c_B.get((w1, w2), 0)
    return (num/den) if den > 0 and num > 0 else 0.0

test_sentence = "students love learning natural language processing"
stoks_B = clean_sentence_to_tokens(test_sentence)

uni_steps_B = [("("+w+")", max(uni_p_mle_B(w), 0.0)) for w in stoks_B]
bi_steps_B  = []
for i in range(len(stoks_B)-1):
    w1, w2 = stoks_B[i], stoks_B[i+1]
    bi_steps_B.append((f"({w1},{w2})", max(bi_p_mle_B(w1, w2), 0.0)))
tri_steps_B = []
for i in range(len(stoks_B)-2):
    w1, w2, w3 = stoks_B[i], stoks_B[i+1], stoks_B[i+2]
    tri_steps_B.append((f"({w1},{w2},{w3})", max(tri_p_mle_B(w1, w2, w3), 0.0)))

def _plot_ngram_steps(title, pairs):
    if not pairs:
        return
    labels = [p[0] for p in pairs]
    vals   = [p[1] for p in pairs]
    eps = 1e-12
    logvals = [math.log10(v + eps) for v in vals]
    plt.figure()
    plt.bar(range(len(labels)), logvals)
    plt.xticks(range(len(labels)), labels, rotation=70)
    plt.ylabel("log10(prob + eps)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

_plot_ngram_steps("Part B — Unigram step probs (unsmoothed)", uni_steps_B)
_plot_ngram_steps("Part B — Bigram step probs (unsmoothed)", bi_steps_B)
_plot_ngram_steps("Part B — Trigram step probs (unsmoothed)", tri_steps_B)

# Part C — Before (MLE) vs After (Laplace) plots on UNK-mapped data
vocab_C = build_vocab(tokens_all, min_freq=2)
tok_unk_C = map_to_unk(tokens_all, vocab_C)
V_C = len(vocab_C)

uni_c_C = ngram_counts(tok_unk_C, 1)
bi_c_C  = ngram_counts(tok_unk_C, 2)
tri_c_C = ngram_counts(tok_unk_C, 3)
N_C = sum(uni_c_C.values())

def laplace_uni_C(w):
    return (uni_c_C.get((w,), 0) + 1) / (N_C + V_C)

def laplace_bi_C(w1, w2):
    return (bi_c_C.get((w1, w2), 0) + 1) / (uni_c_C.get((w1,), 0) + V_C)

def laplace_tri_C(w1, w2, w3):
    return (tri_c_C.get((w1, w2, w3), 0) + 1) / (bi_c_C.get((w1, w2), 0) + V_C)

def norm_C(w):
    return w if w in vocab_C else "<unk>"

# pick some n-grams (likely unseen)
bigram_tests = []
trigram_tests = []
for i in range(len(stoks_B)-1):
    bigram_tests.append((stoks_B[i], stoks_B[i+1]))
for i in range(len(stoks_B)-2):
    trigram_tests.append((stoks_B[i], stoks_B[i+1], stoks_B[i+2]))
bigram_tests += [("quantum", "cat"), ("alice", "students")]
trigram_tests += [("tea", "party", "students"), ("white", "rabbit", "nlp")]

def bi_p_mle_C(w1, w2):
    num = bi_c_B.get((w1, w2), 0)
    den = uni_c_B.get((w1,), 0)
    return (num/den) if den > 0 and num > 0 else 0.0

def tri_p_mle_C(w1, w2, w3):
    num = tri_c_B.get((w1, w2, w3), 0)
    den = bi_c_B.get((w1, w2), 0)
    return (num/den) if den > 0 and num > 0 else 0.0

def plot_before_after_bigrams():
    labels, before, after = [], [], []
    shown = 0
    for w1, w2 in bigram_tests:
        pb = bi_p_mle_C(w1, w2)
        pa = laplace_bi_C(norm_C(w1), norm_C(w2))
        if pb == 0.0 and shown < 5:
            labels.append(f"({w1},{w2})")
            before.append(0.0)
            after.append(pa)
            shown += 1
    if labels:
        plt.figure()
        x = range(len(labels))
        plt.bar([i-0.2 for i in x], before, width=0.4, label="Before (MLE)")
        plt.bar([i+0.2 for i in x], after, width=0.4, label="After (Laplace)")
        plt.xticks(x, labels, rotation=60)
        plt.ylabel("Probability")
        plt.title("Part C — Bigram: Before vs After")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_before_after_trigrams():
    labels, before, after = [], [], []
    shown = 0
    for w1, w2, w3 in trigram_tests:
        pb = tri_p_mle_C(w1, w2, w3)
        pa = laplace_tri_C(norm_C(w1), norm_C(w2), norm_C(w3))
        if pb == 0.0 and shown < 5:
            labels.append(f"({w1},{w2},{w3})")
            before.append(0.0)
            after.append(pa)
            shown += 1
    if labels:
        plt.figure()
        x = range(len(labels))
        plt.bar([i-0.2 for i in x], before, width=0.4, label="Before (MLE)")
        plt.bar([i+0.2 for i in x], after, width=0.4, label="After (Laplace)")
        plt.xticks(x, labels, rotation=60)
        plt.ylabel("Probability")
        plt.title("Part C — Trigram: Before vs After")
        plt.legend()
        plt.tight_layout()
        plt.show()

plot_before_after_bigrams()
plot_before_after_trigrams()

# Part D — Back-off vs Interpolation plots
def backoff_logP(sent_tokens):
    lp = 0.0
    for i in range(len(sent_tokens)-2):
        w1, w2, w3 = map(norm_C, sent_tokens[i:i+3])
        if tri_c_C.get((w1, w2, w3), 0) > 0:
            p = laplace_tri_C(w1, w2, w3)
        elif bi_c_C.get((w2, w3), 0) > 0:
            p = laplace_bi_C(w2, w3)
        else:
            p = laplace_uni_C(w3)
        lp += math.log(p)
    return lp

def interpolation_prob_C(w1, w2, w3, l1=0.6, l2=0.3, l3=0.1):
    w1, w2, w3 = norm_C(w1), norm_C(w2), norm_C(w3)
    return l1*laplace_tri_C(w1, w2, w3) + l2*laplace_bi_C(w2, w3) + l3*laplace_uni_C(w3)

def interpolation_logP(sent_tokens, l1=0.6, l2=0.3, l3=0.1):
    lp = 0.0
    for i in range(len(sent_tokens)-2):
        w1, w2, w3 = sent_tokens[i:i+3]
        lp += math.log(interpolation_prob_C(w1, w2, w3, l1, l2, l3))
    return lp

test_sentences = [
    "students love learning natural language processing",
    "alice was beginning to get very tired",
    "the white rabbit was late"
]

labels_D, backoff_vals, interp_vals = [], [], []
for s in test_sentences:
    toks = clean_sentence_to_tokens(s)
    labels_D.append(s[:28] + ("..." if len(s) > 28 else ""))
    backoff_vals.append(backoff_logP(toks))
    interp_vals.append(interpolation_logP(toks))

plt.figure()
x = range(len(labels_D))
plt.bar([i-0.2 for i in x], backoff_vals, width=0.4, label="Back-off (logP)")
plt.bar([i+0.2 for i in x], interp_vals, width=0.4, label="Interpolation (logP)")
plt.xticks(x, labels_D, rotation=0)
plt.ylabel("log probability (higher is better)")
plt.title("Part D — Back-off vs Interpolation (logP)")
plt.legend()
plt.tight_layout()
plt.show()

# Part E — Perplexity across models/orders
split = int(0.8 * len(tokens_all))
train_raw = tokens_all[:split]
test_raw  = tokens_all[split:]

vocab_E = build_vocab(train_raw, min_freq=2)
train = map_to_unk(train_raw, vocab_E)
test  = map_to_unk(test_raw,  vocab_E)
V_E = len(vocab_E)

uni_c_E = ngram_counts(train, 1)
bi_c_E  = ngram_counts(train, 2)
tri_c_E = ngram_counts(train, 3)
N_E = sum(uni_c_E.values())

def laplace_uni_E(w):
    return (uni_c_E.get((w,), 0) + 1) / (N_E + V_E)

def laplace_bi_E(w1, w2):
    return (bi_c_E.get((w1, w2), 0) + 1) / (uni_c_E.get((w1,), 0) + V_E)

def laplace_tri_E(w1, w2, w3):
    return (tri_c_E.get((w1, w2, w3), 0) + 1) / (bi_c_E.get((w1, w2), 0) + V_E)

def mle_uni_E(w):
    return uni_c_E.get((w,), 0)/N_E if N_E > 0 and uni_c_E.get((w,), 0) > 0 else 0.0

def mle_bi_E(w1, w2):
    num = bi_c_E.get((w1, w2), 0)
    den = uni_c_E.get((w1,), 0)
    return (num/den) if den > 0 and num > 0 else 0.0

def mle_tri_E(w1, w2, w3):
    num = tri_c_E.get((w1, w2, w3), 0)
    den = bi_c_E.get((w1, w2), 0)
    return (num/den) if den > 0 and num > 0 else 0.0

def perplexity_stream(toks, prob_fn, order):
    lp = 0.0
    m = 0
    if order == 1:
        for w in toks:
            p = prob_fn(w)
            if p <= 0.0: return float("inf")
            lp += math.log(p); m += 1
    elif order == 2:
        for i in range(len(toks)-1):
            w1, w2 = toks[i], toks[i+1]
            p = prob_fn(w1, w2)
            if p <= 0.0: return float("inf")
            lp += math.log(p); m += 1
    elif order == 3:
        for i in range(len(toks)-2):
            w1, w2, w3 = toks[i], toks[i+1], toks[i+2]
            p = prob_fn(w1, w2, w3)
            if p <= 0.0: return float("inf")
            lp += math.log(p); m += 1
    if m == 0: return float("inf")
    return math.exp(-lp/m)

# Interpolation (bi with uni; tri with bi+uni)
def interp_bi_prob_E(w1, w2, l1=0.7, l2=0.3):
    return l1*laplace_bi_E(w1, w2) + l2*laplace_uni_E(w2)

def interp_tri_prob_E(w1, w2, w3, l1=0.6, l2=0.3, l3=0.1):
    return l1*laplace_tri_E(w1, w2, w3) + l2*laplace_bi_E(w2, w3) + l3*laplace_uni_E(w3)

pp = {
    "Unigram MLE":        perplexity_stream(test, mle_uni_E, 1),
    "Unigram Laplace":    perplexity_stream(test, laplace_uni_E, 1),
    "Bigram MLE":         perplexity_stream(test, mle_bi_E, 2),
    "Bigram Laplace":     perplexity_stream(test, laplace_bi_E, 2),
    "Bigram Interp":      perplexity_stream(test, lambda w1,w2: interp_bi_prob_E(w1,w2), 2),
    "Trigram MLE":        perplexity_stream(test, mle_tri_E, 3),
    "Trigram Laplace":    perplexity_stream(test, laplace_tri_E, 3),
    "Trigram Interp":     perplexity_stream(test, lambda w1,w2,w3: interp_tri_prob_E(w1,w2,w3), 3),
}

labels_E = list(pp.keys())
vals_E   = [pp[k] for k in labels_E]

# Plot finite perplexities
finite_pairs = [(k, v) for k, v in zip(labels_E, vals_E) if math.isfinite(v)]
infinite     = [(k, v) for k, v in zip(labels_E, vals_E) if not math.isfinite(v)]

if finite_pairs:
    plt.figure()
    x = range(len(finite_pairs))
    plt.bar(x, [v for _, v in finite_pairs])
    plt.xticks(x, [k for k, _ in finite_pairs], rotation=30, ha="right")
    plt.ylabel("Perplexity (lower is better)")
    plt.title("Part E — Perplexity by Model/Order")
    plt.tight_layout()
    plt.show()

if infinite:
    print("\n[NOTE] Infinite perplexities (zero-probability issue):")
    for k, _ in infinite:
        print("  -", k)

print("\nPerplexity summary:")
for k in labels_E:
    print(f"  {k:18s}: {vals_E[labels_E.index(k)]}")
