# -*- coding: utf-8 -*-
import os, re, pickle, numpy as np, tensorflow as tf
from nltk.corpus import gutenberg
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, BatchNormalization, SpatialDropout1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# -----------------------------
# Config
# -----------------------------
VOCAB_SIZE        = 3000          # limit vocab (rest -> <UNK>)
OOV_TOKEN         = "<UNK>"
MAX_SEQ_LEN       = 20            # rolling context length (tokens)
EMBED_DIM         = 100           # 50–128 is reasonable; 100 is a good balance
GRU_UNITS         = 128           # 64–128 is good for this data size
BATCH_SIZE        = 64
EPOCHS            = 80
LR                = 2e-3          # a bit higher to push early learning
L2_WEIGHT         = 1e-5          # gentle L2 to avoid over-regularization
LABEL_SMOOTH      = 0.05
VAL_SPLIT         = 0.2
MODEL_DIR         = "models"
TOKENIZER_PATH    = "pipelines/tokenizer.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)

# -----------------------------
# Load & Clean Text
# -----------------------------
raw = gutenberg.raw('shakespeare-hamlet.txt')

def clean_text(t: str) -> str:
    # to lower
    t = t.lower()

    # remove Gutenberg boilerplate if present (simple heuristic)
    t = re.sub(r'\*{3}.*?\*{3}', ' ', t, flags=re.S)  # *** START/END OF THE PROJECT GUTENBERG EBOOK ***

    # remove stage directions in [ ... ] and ( ... )
    t = re.sub(r'\[.*?\]', ' ', t)
    t = re.sub(r'\(.*?\)', ' ', t)

    # remove speaker tags like "FRANCISCO.", "KING.", "HAMLET." at line starts
    t = re.sub(r'(?m)^\s*[a-z][a-z\- ]{2,}\.\s*$', ' ', t)  # after lowercasing, speakers become lowercase
    # some editions have uppercase lines; as a fallback, drop single-word lines
    t = re.sub(r'(?m)^\s*[a-z]+\.\s*$', ' ', t)

    # keep letters, apostrophes and whitespace; remove the rest
    t = re.sub(r"[^a-z'\s]", " ", t)

    # collapse multiple apostrophes and spaces
    t = re.sub(r"'+", "'", t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

text = clean_text(raw)

print("Sample cleaned text:\n", text[:300], "...\n")

# -----------------------------
# Tokenizer (limited vocab + OOV)
# -----------------------------
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN, filters='')  # filters='' since we already cleaned
tokenizer.fit_on_texts([text])

# effective vocab actually used by Keras embedding
effective_vocab = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)
print(f"Total unique words (raw): {len(tokenizer.word_index)} | Using top: {effective_vocab}")

with open(TOKENIZER_PATH, 'wb') as f:
    pickle.dump(tokenizer, f)

# -----------------------------
# Build rolling n-gram sequences (efficient)
#   Instead of O(n^2) all-prefixes, use a sliding window of MAX_SEQ_LEN
# -----------------------------
tokens = tokenizer.texts_to_sequences([text])[0]
print(f"Total tokens after cleaning: {len(tokens)}")

sequences = []
for i in range(1, len(tokens)):
    start = max(0, i - MAX_SEQ_LEN)
    seq = tokens[start:i+1]  # includes current token as label
    sequences.append(seq)

# Pad sequences to fixed length (MAX_SEQ_LEN + 1 because we include label token)
max_len = MAX_SEQ_LEN + 1
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

# Split into X (all but last) and y (last)
X = sequences[:, :-1]
y_int = sequences[:, -1]  # integer labels
# One-hot with the effective vocab size
y = tf.keras.utils.to_categorical(y_int, num_classes=effective_vocab)

print("X shape:", X.shape, "| y shape:", y.shape, "| Max sequence len:", max_len)

# -----------------------------
# Model (compact, regularized)
#   - No recurrent_dropout (often hurts GRU throughput/generalization)
#   - SpatialDropout on embeddings
#   - Gentle L2
#   - BatchNorm
#   - Label smoothing + gradient clipping
# -----------------------------
model = Sequential()
model.add(Embedding(input_dim=effective_vocab, output_dim=EMBED_DIM, input_length=max_len-1))
model.add(SpatialDropout1D(0.3))

model.add(GRU(
    GRU_UNITS,
    return_sequences=False,
    dropout=0.3,
    kernel_regularizer=l2(L2_WEIGHT),
    recurrent_regularizer=l2(L2_WEIGHT)
))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(effective_vocab, activation='softmax'))

optimizer = Adam(learning_rate=LR, clipnorm=1.0)
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
model.summary()

# -----------------------------
# Callbacks
# -----------------------------
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=12,                 # give LM time; it's noisy
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    mode="min",
    verbose=1,
)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X, y,
    validation_split=VAL_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# -----------------------------
# (Optional) Quick generation helper
# -----------------------------
def generate_text(seed_text, num_words=20, temperature=1.0):
    """Greedy-ish sampling with temperature."""
    result = []
    seq_len = max_len - 1

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-seq_len:]  # keep last context
        token_list = pad_sequences([token_list], maxlen=seq_len, padding='pre')

        preds = model.predict(token_list, verbose=0)[0]
        preds = np.asarray(preds).astype('float64')

        # temperature scaling
        preds = np.log(np.maximum(preds, 1e-9)) / max(temperature, 1e-5)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        next_index = np.random.choice(len(preds), p=preds)
        next_word = tokenizer.index_word.get(next_index, OOV_TOKEN)
        seed_text += ' ' + next_word
        result.append(next_word)

    return seed_text

# Example:
# print(generate_text("to be", num_words=15, temperature=0.8))
