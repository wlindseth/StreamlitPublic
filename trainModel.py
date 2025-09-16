# train_model.py
# Purpose: Train a PyTorch MLP on 20 Newsgroups data and save artifacts.

import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader
import joblib # For saving the vectorizer

# ---- Reproducibility ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# 1. Load Dataset
# =========================
print("Loading 20 Newsgroups dataset...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X_raw, y = data.data, data.target
num_classes = len(data.target_names)
print(f"Dataset loaded with {len(X_raw)} documents and {num_classes} categories.")

# =========================
# 2. Convert Text Data to Numerical Format (TF-IDF)
# =========================
print("Vectorizing text data with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    lowercase=True,
    stop_words='english',
    strip_accents='unicode',
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)
X_vec = vectorizer.fit_transform(X_raw)
X_vec = X_vec.toarray()

# Save the fitted vectorizer
joblib.dump(vectorizer, 'vectorizer.joblib')
print("TF-IDF vectorizer fitted and saved to 'vectorizer.joblib'.")

# =========================
# 3. Split data and create Torch Dataloaders
# =========================
print("Splitting data and creating PyTorch DataLoaders...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=SEED
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

# =========================
# 4. Design Neural Network Architecture
# =========================
class NewsMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Define layers for the neural network
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Forward pass through the network
        return self.net(x)

input_dim = X_train_t.shape[1]
model = NewsMLP(input_dim=input_dim, num_classes=num_classes).to(device)
print("\nModel Architecture:")
print(model)

# =========================
# 5. Define Optimizer and Loss Function
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# =========================
# 6. Train the Model
# =========================
def train(num_epochs=10):
    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss, running_correct, total = 0.0, 0, 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Standard training steps
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            # Track metrics
            preds = logits.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            total += xb.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = running_correct / total
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
    print("--- Training Finished ---")

# =========================
# 7. Evaluate the Model
# =========================
def evaluate():
    print("\n--- Starting Evaluation ---")
    model.eval() # Set model to evaluation mode
    all_preds, all_targets = [], []
    
    with torch.no_grad(): # Disable gradient calculation
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    print(f"\nTest Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=data.target_names))
    print("--- Evaluation Finished ---")


if __name__ == "__main__":
    train(num_epochs=10)
    evaluate()
    
    # Save the final trained model state
    torch.save(model.state_dict(), 'news_classifier_model.pth')
    print("\nTrained model state saved to 'news_classifier_model.pth'.")