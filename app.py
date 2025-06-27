import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import joblib
from transformers import pipeline as hf_pipeline
import re

# 1. Load datasets
df = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=14D_HcvTFL63-KffCQLNFxGH-oY_knwmo",
    delimiter=';', header=None, names=['sentence', 'label']
)
ts_df = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=1Vmr1Rfv4pLSlAUrlOCxAcszvlxJOSHrm",
    delimiter=';', header=None, names=['sentence', 'label']
)
df = pd.concat([df, ts_df], ignore_index=True)
df
total_rows = df.shape[0]

# % of null values
null_percent = df.isnull().mean() * 100

# % of duplicate rows
duplicate_rows = df.duplicated().sum()
duplicate_percent = (duplicate_rows / total_rows) * 100

print("Null Value Percentage:\n", null_percent)
print(f"\n📄 Duplicate Rows: {duplicate_rows} ({duplicate_percent:.2f}%)")
df.drop_duplicates(inplace=True)
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)         # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)                         # Remove @ and #
    text = re.sub(r'[^a-z\s]', '', text)                        # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()                    # Normalize spaces
    return text
df['clean_sentence'] = df['sentence'].apply(clean_text)
# Load and prepare data
X = df['clean_sentence']
y = df['label']
# 1. Install necessary libraries in Colab (run once)
!pip install textblob
!python -m textblob.download_corpora
# === MODEL TRAINING CODE WITH REQUIRED CONCEPTS ===

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# --- 1. Load and preprocess your DataFrame ---

tokenized = df['clean_sentence'].apply(str.split)

# --- 2. Build Vocabulary ---
vocab = Counter([token for sentence in tokenized for token in sentence])
vocab = {word: i+2 for i, (word, _) in enumerate(vocab.most_common())}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

def encode(text):
    return [vocab.get(word, vocab['<UNK>']) for word in text]

encoded_texts = tokenized.apply(encode)

# --- 3. Pad Sequences ---
MAX_LEN = 32
def pad_sequence(seq):
    return seq[:MAX_LEN] + [vocab['<PAD>']] * max(0, MAX_LEN - len(seq))
padded = encoded_texts.apply(pad_sequence).tolist()

# --- 4. Encode Labels ---
le = LabelEncoder()
labels = le.fit_transform(df['label'])

# --- 5. Dataset + DataLoader ---
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(padded, labels, test_size=0.2, stratify=labels, random_state=42)
train_loader = DataLoader(EmotionDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(EmotionDataset(X_val, y_val), batch_size=16)

# --- 6. Co-occurrence Matrix (Visualization Only) ---
vectorizer = CountVectorizer(max_features=20)
X_counts = vectorizer.fit_transform(df['clean_sentence'])
X_counts = (X_counts.T * X_counts)
X_counts.setdiag(0)
plt.figure(figsize=(18, 18))
sns.heatmap(X_counts.toarray(), xticklabels=vectorizer.get_feature_names_out(),
            yticklabels=vectorizer.get_feature_names_out(), cmap="YlGnBu", annot=True)
plt.title("Word Co-occurrence Matrix")
plt.show()

# --- 7. Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# --- 8. Transformer Model with Masking + Dropout for Bayesian Inference ---
class EmotionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<PAD>'])
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        mask = (x == vocab['<PAD>'])
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.dropout(x.mean(dim=1))  # mean pooling
        return self.fc(x)

# --- 9. Train the Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionTransformer(len(vocab), embed_dim=64, num_heads=4, num_classes=len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Accuracy: {correct / total:.4f}")

# Save model
torch.save(model.state_dict(), "emotion_transformer_model.pth")
! pip install textblob
! python -m textblob.download_corpora
import torch
import torch.nn.functional as F
import random
from textblob import TextBlob

# Load model
model.load_state_dict(torch.load("emotion_transformer_model.pth", map_location=device))
model.eval()

# Preprocess user input
def preprocess_input(text):
    tokens = text.lower().split()
    encoded = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    padded = encoded[:MAX_LEN] + [vocab['<PAD>']] * max(0, MAX_LEN - len(encoded))
    return torch.tensor([padded], dtype=torch.long).to(device)

# Emotion responses
responses = {
    "sadness": [
        "It’s okay to feel down sometimes. I’m here to support you.",
        "I'm really sorry you're going through this. Want to talk more about it?",
        "You're not alone — I’m here for you."
    ],
    "anger": [
        "That must have been frustrating. Want to vent about it?",
        "It's okay to feel this way. I'm listening.",
        "Would it help to talk through it?"
    ],
    "love": [
        "That’s beautiful to hear! What made you feel that way?",
        "It’s amazing to experience moments like that.",
        "Sounds like something truly meaningful."
    ],
    "happiness": [
        "That's awesome! What’s bringing you joy today?",
        "I love hearing good news. 😊",
        "Yay! Want to share more about it?"
    ],
    "neutral": [
        "Got it. I’m here if you want to dive deeper.",
        "Thanks for sharing that. Tell me more if you’d like.",
        "I’m listening. How else can I support you?"
    ]
}

# Suggestions
relaxation_resources = {
    "exercise": "Try this 5-4-3-2-1 grounding method:\n- 5 things you see\n- 4 you can touch\n- 3 you hear\n- 2 you smell\n- 1 you taste",
    "video": "Here’s a short calming video that might help: https://youtu.be/O-6f5wQXSu8"
}

# Keywords
help_keywords = ["suggest", "help", "calm", "exercise", "relax", "how can i", "any tips", "can u", "can you"]
negative_inputs = ["not good", "feel bad", "feel sad", "anxious", "depressed", "upset", "feel like shit", "stress", "worried"]
thank_you_inputs = ["thank", "thanks", "thank you"]
bye_inputs = ["bye", "goodbye", "see you", "take care", "ok bye", "exit", "quit"]

# Conversation state
awaiting_tip_type = False

# Correct spelling
def correct_spelling(text):
    return str(TextBlob(text).correct())

# Get response
def get_response(emotion, user_input):
    global awaiting_tip_type
    user_input_lower = user_input.lower()

    if any(bye in user_input_lower for bye in bye_inputs):
        return "Take care! I’m here whenever you want to talk. 🌿", True

    if any(thank in user_input_lower for thank in thank_you_inputs):
        return "You're most welcome! I'm really glad I could support you. 💙", False

    # Awaiting video vs exercise clarification
    if awaiting_tip_type:
        if "video" in user_input_lower:
            awaiting_tip_type = False
            return relaxation_resources["video"], False
        elif "exercise" in user_input_lower or "excercise" in user_input_lower or "breathe" in user_input_lower:
            awaiting_tip_type = False
            return relaxation_resources["exercise"], False
        else:
            return "Just checking — would you prefer a calming video or a simple breathing exercise?", False

    # Offer relaxation suggestions
    if any(kw in user_input_lower for kw in help_keywords):
        awaiting_tip_type = True
        return "Would you prefer a short calming video or a simple breathing exercise?", False

    # Default: emotional response
    if emotion in responses:
        return random.choice(responses[emotion]), False
    else:
        return random.choice(responses["neutral"]), False

# Main chatbot loop
print("EmotiBot 🌿: Hi! How are you feeling today? (Type 'exit' to quit)")

while True:
    user_input_raw = input("You: ").strip()
    user_input = correct_spelling(user_input_raw)

    if user_input.lower() in ['exit', 'quit']:
        print("EmotiBot 🌿: Take care! I’m here whenever you want to talk.")
        break

    # Emotion prediction
    if any(phrase in user_input.lower() for phrase in negative_inputs):
        pred_emotion = "sadness"
    else:
        x = preprocess_input(user_input)
        model.train()
        with torch.no_grad():
            probs = torch.stack([F.softmax(model(x), dim=1) for _ in range(5)])
            avg_probs = probs.mean(dim=0)
            pred_idx = torch.argmax(avg_probs, dim=1).item()
        pred_emotion = le.classes_[pred_idx]

    # Generate response
    reply, should_exit = get_response(pred_emotion, user_input)
    print(f"EmotiBot 🌿: {reply}")
    if should_exit:
        break