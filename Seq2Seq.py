# ============================================
# ALL-IN-ONE NOTEBOOK SCRIPT (Colab Ready)
# Seq2Seq (Keras) + Image Captioning (PyTorch)
# ============================================

# =========================
# A) Char-level Seq2Seq (Keras / TF2)
# =========================
print("\n[A] Installing TensorFlow for Seq2Seq…")
!pip -q install tensorflow==2.15.0

import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models

print("\n[A] Building a tiny char-level English→French toy dataset…")
pairs = [
    ("hi.", "salut."),
    ("how are you?", "comment ça va ?"),
    ("i love you.", "je t'aime."),
    ("thank you!", "merci !"),
    ("good morning.", "bonjour."),
    ("good night.", "bonne nuit."),
    ("see you.", "à bientôt."),
    ("i am fine.", "je vais bien."),
    ("where are you?", "où es-tu ?"),
    ("what is this?", "qu'est-ce que c'est ?"),
]
input_texts, target_texts = [], []
for src, tgt in pairs:
    input_texts.append(src.lower())
    target_texts.append("\t" + tgt.lower() + "\n")  # start: '\t', end: '\n'

input_chars = sorted(list(set("".join(input_texts))))
target_chars = sorted(list(set("".join(target_texts))))
num_encoder_tokens = len(input_chars)
num_decoder_tokens = len(target_chars)

input_char_index = {c:i for i,c in enumerate(input_chars)}
target_char_index = {c:i for i,c in enumerate(target_chars)}
reverse_target_char_index = {i:c for c,i in target_char_index.items()}

max_encoder_len = max(len(s) for s in input_texts)
max_decoder_len = max(len(s) for s in target_texts)

print(f"[A] Vocab (src/target): {num_encoder_tokens}/{num_decoder_tokens}")
print(f"[A] Max lens (enc/dec): {max_encoder_len}/{max_decoder_len}")

print("[A] Vectorizing…")
encoder_input_data = np.zeros((len(input_texts), max_encoder_len, num_encoder_tokens), dtype="float32")
decoder_input_data = np.zeros((len(input_texts), max_decoder_len, num_decoder_tokens), dtype="float32")
decoder_target_data = np.zeros((len(input_texts), max_decoder_len, num_decoder_tokens), dtype="float32")
for i, (src, tgt) in enumerate(zip(input_texts, target_texts)):
    for t, ch in enumerate(src):
        encoder_input_data[i, t, input_char_index[ch]] = 1.0
    for t, ch in enumerate(tgt):
        decoder_input_data[i, t, target_char_index[ch]] = 1.0
        if t > 0:
            decoder_target_data[i, t-1, target_char_index[ch]] = 1.0

LATENT = 128
print("[A] Building encoder–decoder…")
encoder_inputs = layers.Input(shape=(None, num_encoder_tokens))
encoder_lstm = layers.LSTM(LATENT, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = layers.Input(shape=(None, num_decoder_tokens))
decoder_lstm  = layers.LSTM(LATENT, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

seq2seq = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
seq2seq.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
seq2seq.summary()

print("[A] Training briefly (toy)…")
seq2seq.fit([encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=8, epochs=100, verbose=0)

print("[A] Building inference models…")
encoder_model = models.Model(encoder_inputs, encoder_states)
dec_state_input_h = layers.Input(shape=(LATENT,))
dec_state_input_c = layers.Input(shape=(LATENT,))
dec_states_inputs  = [dec_state_input_h, dec_state_input_c]
dec_outputs, state_h_i, state_c_i = decoder_lstm(decoder_inputs, initial_state=dec_states_inputs)
dec_outputs = decoder_dense(dec_outputs)
decoder_model = models.Model([decoder_inputs] + dec_states_inputs,
                             [dec_outputs, state_h_i, state_c_i])

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1, num_decoder_tokens), dtype="float32")
    target_seq[0, 0, target_char_index["\t"]] = 1.0
    stop = False
    decoded = []
    while not stop and len(decoded) < max_decoder_len:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_index = np.argmax(output_tokens[0, -1, :])
        char = reverse_target_char_index[sampled_index]
        if char == "\n":
            stop = True
        else:
            decoded.append(char)
            target_seq = np.zeros((1, 1, num_decoder_tokens), dtype="float32")
            target_seq[0, 0, sampled_index] = 1.0
            states_value = [h, c]
    return "".join(decoded)

def vectorize_src(s):
    s = s.lower()
    arr = np.zeros((1, max_encoder_len, num_encoder_tokens), dtype="float32")
    for t,ch in enumerate(s):
        if ch in input_char_index:
            arr[0, t, input_char_index[ch]] = 1.0
    return arr

tests = ["hi.", "good night.", "thank you!", "where are you?"]
print("\n[A] Demo translations:")
for t in tests:
    print(f"  {t}  →  {decode_sequence(vectorize_src(t))}")


# =========================
# B) Image Captioning (PyTorch, inference only)
# =========================
print("\n[B] Installing PyTorch & cloning image captioning tutorial…")
!pip -q install torch torchvision pillow==10.4.0

import os, shutil, urllib.request
from pathlib import Path

!git -q clone https://github.com/yunjey/pytorch-tutorial.git
%cd /content/pytorch-tutorial/tutorials/03-advanced/image_captioning
os.makedirs('data', exist_ok=True)
os.makedirs('sample_images', exist_ok=True)

print("\n[B] Upload encoder/decoder/vocab weights (3 files)…")
# If you already have URLs, you could urllib.request.urlretrieve into data/
from google.colab import files
uploaded = files.upload()  # pick encoder-*.pkl, decoder-*.pkl, vocab.pkl
for name in uploaded:
    shutil.move(f"/content/{name}", f"/content/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/{name}")

test_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
urllib.request.urlretrieve(test_url, "sample_images/dog.jpg")

print("\n[B] Running sample.py (edit paths if your filenames differ)…")
!python sample.py \
  --image sample_images/dog.jpg \
  --encoder_path data/encoder-5-3000.pkl \
  --decoder_path data/decoder-5-3000.pkl \
  --vocab_path data/vocab.pkl \
  --embed_size 256 --hidden_size 512 --num_layers 1


# =========================
# C) Porting PyTorch captioner → Keras (notes)
# =========================
"""
[C] Porting steps (summary):

1) Architecture parity
   - Encoder: same CNN backbone (e.g., ResNet-50) with include_top=False, pooling='avg'
   - Dense projection to embed_size
   - Decoder: Embedding(vocab, embed_dim) → LSTM(hidden) → Dense(vocab)
   - Match embed_size, hidden_size, num_layers.

2) Weight mapping
   - Embedding: same shape [vocab, embed_dim] in both frameworks.
   - LSTM gates:
       PyTorch splits parameters by gate (W_ih, W_hh, b_ih, b_hh) ordered [i, f, g, o]
       Keras expects concatenated kernels/recurrent_kernels in [i, f, c, o] by default.
       Slice and concatenate accordingly before set_weights([...]).
   - Dense/Linear: transpose PyTorch weight [out, in] → Keras [in, out].

3) Export
   - Load .pkl/.pth in torch, .cpu().numpy() for arrays,
     re-order gates, then save .npz; in Keras, layer.set_weights([W, b]).

4) Tokenization
   - Reuse the same vocab.pkl indices and special tokens (<start>, <end>, <pad>, <unk>).

5) Inference
   - Implement the same greedy/beam loop in Keras to match outputs.
"""


# =========================
# D) Minimal Keras captioner (encoder/decoder skeleton)
# =========================
print("\n[D] Building minimal Keras captioner (skeleton models)…")

import tensorflow as tf
from tensorflow.keras import layers, models

def build_encoder(embed_dim=256, cnn="resnet50"):
    if cnn == "resnet50":
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg")
        for l in base.layers:
            l.trainable = False
        features = layers.Input(shape=(224,224,3))
        x = tf.keras.applications.resnet50.preprocess_input(features)
        x = base(x)                # (None, 2048)
        x = layers.Dense(embed_dim, activation=None, name="enc_proj")(x)  # (None, embed_dim)
        return models.Model(features, x, name="Encoder")

def build_decoder(vocab_size, embed_dim=256, hidden=512):
    img_feat = layers.Input(shape=(embed_dim,), name="img_feat")
    seq_in   = layers.Input(shape=(None,), name="seq_in")   # int ids
    emb = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="embed")(seq_in)
    img_step = layers.Reshape((1, embed_dim))(img_feat)
    x = layers.Concatenate(axis=1)([img_step, emb])
    x = layers.LSTM(hidden, return_sequences=True, name="lstm")(x)
    logits = layers.TimeDistributed(layers.Dense(vocab_size), name="logits")(x)
    return models.Model([img_feat, seq_in], logits, name="Decoder")

# Example model builds (use your real vocab size)
vocab_size = 10000
enc = build_encoder(embed_dim=256)
dec = build_decoder(vocab_size, embed_dim=256, hidden=512)
enc.summary()
dec.summary()

"""
Training sketch:
- Prepare (image, caption_in, caption_out) mini-batches
- caption_in starts with <start>, caption_out is shifted to include <end>
- loss = SparseCategoricalCrossentropy(from_logits=True), mask padding
- Optionally fine-tune part of the CNN later with small LR
"""


# =========================
# E) Short research notes (as comments)
# =========================
"""
[E1] Extending Seq2Seq to other languages:
- Prefer subwords (SentencePiece/BPE) over characters/words to balance vocab size and OOV handling.
- Use Attention (Bahdanau/Luong) or Transformer (Vaswani et al.) for better MT.
- Evaluate with BLEU/chrF/COMET; apply beam search + length penalty.

[E2] Advanced MT methods:
- Transformers, pretraining (mBART, mT5), multilingual (M2M-100, NLLB).
- Back-translation, label smoothing, knowledge distillation.

[E3] Text → Image generation (inverse of captioning):
- Diffusion models: Stable Diffusion / SDXL / Imagen / DALL·E 2/3.
- Components: text encoder (CLIP), UNet in latent space, schedulers, CFG.

[E4] Char vs char_wb in CountVectorizer:
- 'char': character n-grams across whitespace
- 'char_wb': within-word only, ignores cross-word n-grams
"""

print("\nAll sections finished. For captioning, remember to provide the correct weight filenames.")
