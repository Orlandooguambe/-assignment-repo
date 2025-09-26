# ============================================
# NLP Sprint — IMDB: BoW, TF-IDF, Word2Vec
# Complete, Colab-ready, English comments
# ============================================

!pip -q install gensim==4.3.2 nltk==3.9.1 scikit-learn==1.5.2

import os, re, tarfile, glob, math, random, urllib.request, zipfile
import numpy as np
import pandas as pd
from pathlib import Path

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from gensim.models import Word2Vec

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 4)
np.random.seed(0); random.seed(0)

# --------------------------------------------
# 0) Download IMDB (aclImdb) and prepare
# --------------------------------------------
DATA_DIR = Path('/content/aclImdb')
if not DATA_DIR.exists():
    print("Downloading IMDB dataset...")
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    local = "/content/aclImdb_v1.tar.gz"
    urllib.request.urlretrieve(url, local)
    with tarfile.open(local, 'r:gz') as tfz:
        tfz.extractall('/content')
    # remove unlabeled set (not needed)
    !rm -rf /content/aclImdb/train/unsup

print("IMDB directory structure ready.")
!head -n 20 /content/aclImdb/README

# Load with sklearn.load_files (keeps raw text)
train_data = load_files("/content/aclImdb/train/", encoding="utf-8")
test_data  = load_files("/content/aclImdb/test/",  encoding="utf-8")
x_train, y_train = train_data.data, train_data.target
x_test,  y_test  = test_data.data,  test_data.target
print("Labels:", train_data.target_names)
print("Train / Test sizes:", len(x_train), len(x_test))
print("Sample doc:\n", x_train[0][:400], "...")

# =====================================================
# 1) BoW scratch (unigram + bigram) on 3 small strings
# =====================================================
mini_dataset = [
    "This movie is SOOOO funny!!!!",
    "What a movie! I never",
    "best movie ever!!!!! this movie",
]

def simple_tokenize(text):
    # Lowercase, remove URLs and non-alphanum (keep spaces), split
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

def build_vocab_unigram(sentences):
    vocab = {}
    for s in sentences:
        for tok in simple_tokenize(s):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab  # {token: index}

def build_vocab_ngram(sentences, n=2):
    vocab = {}
    for s in sentences:
        toks = simple_tokenize(s)
        ngrams = [' '.join(toks[i:i+n]) for i in range(len(toks)-n+1)]
        for ng in ngrams:
            if ng not in vocab:
                vocab[ng] = len(vocab)
    return vocab

def bow_counts(sentences, vocab, n=1):
    X = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, s in enumerate(sentences):
        toks = simple_tokenize(s)
        units = toks if n==1 else [' '.join(toks[j:j+n]) for j in range(len(toks)-n+1)]
        for u in units:
            if u in vocab:
                X[i, vocab[u]] += 1
    return X

# Unigram
uni_vocab = build_vocab_unigram(mini_dataset)
X_uni = bow_counts(mini_dataset, uni_vocab, n=1)
print("\n[BoW Scratch] Unigram vocabulary:", uni_vocab)
print(pd.DataFrame(X_uni, columns=[k for k,_ in sorted(uni_vocab.items(), key=lambda x:x[1])]))

# Bigram
bi_vocab = build_vocab_ngram(mini_dataset, n=2)
X_bi = bow_counts(mini_dataset, bi_vocab, n=2)
print("\n[BoW Scratch] Bigram vocabulary:", bi_vocab)
print(pd.DataFrame(X_bi, columns=[k for k,_ in sorted(bi_vocab.items(), key=lambda x:x[1])]))

# =====================================================
# 2) TF-IDF vectorization on IMDB (stopwords + max vocab)
# =====================================================
english_sw = stopwords.words('english')
MAX_FEAT = 5000

tfidf = TfidfVectorizer(
    max_features=MAX_FEAT,
    stop_words=english_sw,
    token_pattern=r'(?u)\b\w+\b',  # include 1-char tokens
    norm=None                      # no L2 normalization (per assignment note)
)
Xtr_tfidf = tfidf.fit_transform(x_train)
Xte_tfidf = tfidf.transform(x_test)

print("\nTF-IDF shapes:", Xtr_tfidf.shape, Xte_tfidf.shape)
print("Sample TF-IDF vocab size:", len(tfidf.vocabulary_))

# =====================================================
# 3) Train any binary classifier on TF-IDF (accuracy report)
# =====================================================
clf = LogisticRegression(max_iter=200, n_jobs=None, solver='liblinear')
clf.fit(Xtr_tfidf, y_train)
y_pred = clf.predict(Xte_tfidf)
acc = accuracy_score(y_test, y_pred)
print("\n[TF-IDF → LogisticRegression] Test accuracy:", f"{acc:.4f}")
print(classification_report(y_test, y_pred, target_names=train_data.target_names))

# =====================================================
# 4) TF-IDF scratch (no sklearn) on 3 small strings
#     (a) Standard: tf = count/len(doc), idf = ln(N/df)
#     (b) sklearn-like: tf = count, idf = ln( (1+N)/(1+df) ) + 1
#     No normalization in both.
# =====================================================
def scratch_tfidf(docs, variant='standard'):
    # Build unigram vocab over docs
    vocab = build_vocab_unigram(docs)
    N = len(docs)
    counts = np.zeros((N, len(vocab)), dtype=float)
    df = np.zeros(len(vocab), dtype=int)

    # term counts and doc frequency
    for i, s in enumerate(docs):
        toks = simple_tokenize(s)
        for t in toks:
            j = vocab[t]
            counts[i, j] += 1
        # df: how many docs contain token
        for j in set([vocab[t] for t in toks]):
            df[j] += 1

    # compute TF and IDF according to variant
    if variant == 'standard':
        # tf = count / sum(counts in doc), idf = ln(N / df)
        tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
        idf = np.log(np.maximum(N / np.maximum(df, 1), 1e-12))
        tfidf = tf * idf
    elif variant == 'sklearn':
        # tf = raw count, idf = ln((1+N)/(1+df)) + 1
        tf = counts
        idf = np.log((1 + N) / (1 + df)) + 1.0
        tfidf = tf * idf
    else:
        raise ValueError("variant must be 'standard' or 'sklearn'")
    inv_vocab = [k for k,_ in sorted(vocab.items(), key=lambda x:x[1])]
    return tfidf, inv_vocab

docs3 = [
    "This movie is SOOOO funny!!!",
    "What a movie! I never",
    "best movie ever!!!!! this movie",
]

tfidf_std, vocab_std = scratch_tfidf(docs3, variant='standard')
tfidf_skl, vocab_skl = scratch_tfidf(docs3, variant='sklearn')

print("\n[TF-IDF Scratch] Standard formula")
print(pd.DataFrame(tfidf_std, columns=vocab_std).round(4))

print("\n[TF-IDF Scratch] scikit-learn-like formula")
print(pd.DataFrame(tfidf_skl, columns=vocab_skl).round(4))

# =====================================================
# 5) Corpus preprocessing for Word2Vec (IMDB train)
#    - lowercase, remove URLs & non-alphanum, split into tokens
# =====================================================
def clean_tokenize(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

print("\nTokenized sample:", clean_tokenize(x_train[0])[:20])

# =====================================================
# 6) Train Word2Vec on IMDB training corpus
# =====================================================
# For speed, we sample some docs; feel free to set to None to use all.
SAMPLE_DOCS = 10000  # None for full ~25k
train_sample = x_train if SAMPLE_DOCS is None else x_train[:SAMPLE_DOCS]

sentences = [clean_tokenize(doc) for doc in train_sample]
# Filter empty after cleaning
sentences = [s for s in sentences if len(s) > 0]

EMBED_DIM = 100  # vector size for Word2Vec
w2v = Word2Vec(
    sentences=sentences,
    vector_size=EMBED_DIM,
    window=5, min_count=2, workers=2, sg=1,  # sg=1 (skip-gram), sg=0 for CBOW
    epochs=5
)
print("\nWord2Vec vocab size:", len(w2v.wv))

# =====================================================
# 7) (Advanced) t-SNE visualization and most_similar
# =====================================================
RUN_TSNE = True
if RUN_TSNE:
    # Pick top-N frequent words
    topN = 200
    words = w2v.wv.index_to_key[:topN]
    vectors = np.stack([w2v.wv[w] for w in words], axis=0)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1500, random_state=0)
    emb2d = tsne.fit_transform(vectors)
    plt.figure(figsize=(8,6))
    plt.scatter(emb2d[:,0], emb2d[:,1], s=10)
    for i, w in enumerate(words[:80]):  # annotate a subset to keep it readable
        plt.annotate(w, (emb2d[i,0], emb2d[i,1]), fontsize=8, alpha=0.7)
    plt.title("t-SNE of Word2Vec embeddings (subset)")
    plt.show()

    for probe in ["good", "bad", "movie", "plot"]:
        if probe in w2v.wv:
            print(f"Most similar to '{probe}':", w2v.wv.most_similar(probe, topn=5))
        else:
            print(f"'{probe}' not in vocab.")

# =====================================================
# 8) Classification with learned and pre-trained embeddings
#    Approach: Average word vectors per review -> LogisticRegression
# =====================================================

def average_embeddings(docs, wv, unk=None):
    """
    docs: list of strings
    wv: gensim KeyedVectors or dict-like
    unk: if provided (np.array), use when no token found; else zeros
    """
    vecs = np.zeros((len(docs), EMBED_DIM), dtype=np.float32)
    for i, doc in enumerate(docs):
        toks = clean_tokenize(doc)
        acc = []
        for t in toks:
            if t in wv:
                acc.append(wv[t])
        if len(acc) == 0:
            vecs[i] = unk if unk is not None else 0.0
        else:
            vecs[i] = np.mean(acc, axis=0)
    return vecs

# 8A) Using our trained Word2Vec
Xtr_w2v = average_embeddings(x_train, w2v.wv)
Xte_w2v = average_embeddings(x_test,  w2v.wv)

clf_w2v = LogisticRegression(max_iter=300, solver='liblinear')
clf_w2v.fit(Xtr_w2v, y_train)
pred_w2v = clf_w2v.predict(Xte_w2v)
acc_w2v = accuracy_score(y_test, pred_w2v)
print("\n[Avg Word2Vec (trained)] Test accuracy:", f"{acc_w2v:.4f}")

# 8B) Using pre-trained GloVe 100d
RUN_GLOVE = True
if RUN_GLOVE:
    glove_dir = Path("/content/glove.6B")
    glove_txt = glove_dir/"glove.6B.100d.txt"
    if not glove_txt.exists():
        print("Downloading GloVe 6B (822 MB zip)...")
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = "/content/glove.6B.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(glove_dir)

    print("Loading GloVe 100d...")
    glove = {}
    with open(glove_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            w = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            glove[w] = vec
    print("GloVe loaded, vocab:", len(glove))

    # Align dimensions to 100 for GloVe section
    EMBED_DIM = 100
    def avg_glove(docs, glove):
        vecs = np.zeros((len(docs), EMBED_DIM), dtype=np.float32)
        for i, doc in enumerate(docs):
            toks = clean_tokenize(doc)
            acc = [glove[t] for t in toks if t in glove]
            if len(acc):
                vecs[i] = np.mean(acc, axis=0)
        return vecs

    Xtr_gl = avg_glove(x_train, glove)
    Xte_gl = avg_glove(x_test,  glove)
    clf_gl = LogisticRegression(max_iter=300, solver='liblinear')
    clf_gl.fit(Xtr_gl, y_train)
    pred_gl = clf_gl.predict(Xte_gl)
    acc_gl = accuracy_score(y_test, pred_gl)
    print("[Avg GloVe 100d] Test accuracy:", f"{acc_gl:.4f}")

print("\nDone.")
