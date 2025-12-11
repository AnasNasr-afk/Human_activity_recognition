# HAC using smart phones

This notebook documents a complete ML workflow for Human Activity Recognition using the **UCI HAR Dataset**.

It includes:

1. Data loading from raw `.txt` files
2. Preprocessing:
    - Train/Validation/Test split
    - Scaling
    - Label Encoding
    - PCA (95%)
3. Model Training (LR & RF)
4. Evaluation across **train, validation, and test**
5. Overfitting/underfitting diagnostics
6. Learning curves

---

---

# 📂 **1. Load Dataset (From UCI HAR)**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, log_loss

# Load X/y from training set
df_X_temp = pd.read_csv(
    r"/Users/anasnasr/Library/CloudStorage/OneDrive-FutureUniversityinEgypt/UCI HAR Dataset/train/X_train.txt",
    sep=r'\s+',
    header=None
)

df_y_temp = pd.read_csv(
    r"/Users/anasnasr/Library/CloudStorage/OneDrive-FutureUniversityinEgypt/UCI HAR Dataset/train/y_train.txt",
    sep=r'\s+',
    header=None
).squeeze()

# Load X/y from testing set
df_X_test = pd.read_csv(
    r"/Users/anasnasr/Library/CloudStorage/OneDrive-FutureUniversityinEgypt/UCI HAR Dataset/test/X_test.txt",
    sep=r'\s+',
    header=None
)

df_y_test = pd.read_csv(
    r"/Users/anasnasr/Library/CloudStorage/OneDrive-FutureUniversityinEgypt/UCI HAR Dataset/test/y_test.txt",
    sep=r'\s+',
    header=None
).squeeze()

```

---

# 🧹 **2. Preprocessing**

## 2.1 **Train/Validation Split**

```python
X_train, X_val, y_train, y_val = train_test_split(
    df_X_temp, df_y_temp,
    test_size=0.2,
    random_state=42,
    stratify=df_y_temp
)

```

---

## 2.2 **Scaling — Fit Only on Training Data**

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(df_X_test)

```

---

## 2.3 **Label Encoding — Fit Only on Training Data**

```python
label_encoder = LabelEncoder()

y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc   = label_encoder.transform(y_val)
y_test_enc  = label_encoder.transform(df_y_test)

print("Classes:", label_encoder.classes_)

```

---

## 2.4 **PCA Reduction (95% Variance)**

```python
pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print("Original features:", X_train.shape[1])
print("PCA components   :", X_train_pca.shape[1])

```

### PCA Explained Variance Plot

```python
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA – Explained Variance")
plt.grid(True)
plt.show()

```

---

# 🤖 **3. Training + Evaluation + Overfitting Analysis**

```python
def train_and_evaluate(name, model):
    print(f"\n{'='*70}")
    print(f"Training {name}...")
    print('='*70)

    model.fit(X_train_pca, y_train_enc)
    print("✓ Model trained successfully")

    # Predictions
    y_train_pred = model.predict(X_train_pca)
    y_val_pred   = model.predict(X_val_pca)
    y_test_pred  = model.predict(X_test_pca)

    y_train_proba = model.predict_proba(X_train_pca)
    y_val_proba   = model.predict_proba(X_val_pca)
    y_test_proba  = model.predict_proba(X_test_pca)

    # Metrics
    acc_train = accuracy_score(y_train_enc, y_train_pred)
    acc_val   = accuracy_score(y_val_enc,   y_val_pred)
    acc_test  = accuracy_score(y_test_enc,  y_test_pred)

    prec_train = precision_score(y_train_enc, y_train_pred, average="macro")
    prec_val   = precision_score(y_val_enc,   y_val_pred, average="macro")
    prec_test  = precision_score(y_test_enc,  y_test_pred, average="macro")

    loss_train = log_loss(y_train_enc, y_train_proba)
    loss_val   = log_loss(y_val_enc,   y_val_proba)
    loss_test  = log_loss(y_test_enc,  y_test_proba)

    print(f"\n{name} TRAIN Results:")
    print(f"  Accuracy : {acc_train:.6f}")
    print(f"  Precision: {prec_train:.6f}")
    print(f"  Loss     : {loss_train:.6f}")

    print(f"\n{name} VALIDATION Results:")
    print(f"  Accuracy : {acc_val:.6f}")
    print(f"  Precision: {prec_val:.6f}")
    print(f"  Loss     : {loss_val:.6f}")

    print(f"\n{name} TEST Results:")
    print(f"  Accuracy : {acc_test:.6f}")
    print(f"  Precision: {prec_test:.6f}")
    print(f"  Loss     : {loss_test:.6f}")

    # ----------------------------
    # Overfitting Analysis
    # ----------------------------
    print(f"\n{name} OVERFITTING ANALYSIS:")
    print("-"*70)

    acc_gap_val  = acc_train - acc_val
    acc_gap_test = acc_train - acc_test

    loss_gap_val  = loss_val - loss_train
    loss_gap_test = loss_test - loss_train

    print(f"Train vs Validation Accuracy Gap: {acc_gap_val:+.6f}")
    print(f"Train vs Test Accuracy Gap      : {acc_gap_test:+.6f}")

    print(f"Validation Loss Gap (val - train): {loss_gap_val:+.6f}")
    print(f"Test Loss Gap (test - train)     : {loss_gap_test:+.6f}")

    if acc_train < 0.85:
        print("⚠️ UNDERFITTING: training accuracy < 0.85")

    # ----------------------------
    # Learning Curve
    # ----------------------------
    print(f"\nGenerating learning curve for {name}...")

    params = model.get_params()
    if "n_jobs" in params:
        params["n_jobs"] = 1

    model_curve = model.__class__(**params)

    train_sizes, train_scores, val_scores = learning_curve(
        model_curve,
        X_train_pca, y_train_enc,
        train_sizes=np.linspace(0.3, 1.0, 4),
        cv=2,
        scoring="neg_log_loss",
        shuffle=True,
        random_state=42,
        n_jobs=1
    )

    train_loss = -train_scores.mean(axis=1)
    val_loss   = -val_scores.mean(axis=1)

    plt.figure(figsize=(10,6))
    plt.plot(train_sizes, train_loss, marker="o", label="Training Loss")
    plt.plot(train_sizes, val_loss, marker="s", label="Validation Loss")
    plt.xlabel("Training Samples")
    plt.ylabel("Log Loss")
    plt.title(f"Learning Curve – {name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("✓ Learning curve generated")

```

---

# 🌟 **4. Model – Logistic Regression**

```python
log_reg = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

train_and_evaluate("Logistic Regression", log_reg)

```

---

# 🌲 **5. Model – Random Forest**

```python
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

train_and_evaluate("Random Forest", rf)

```

---

# 🏁 **6. Final Summary**

✓ Loaded UCI HAR dataset from raw text files

✓ Preprocessed using scaling, encoding, PCA

✓ Split into train, validation, test

✓ Trained LR & RandomForest

✓ Evaluated full metrics: **train/val/test**

✓ Overfitting and underfitting analysis

✓ Learning curves for model behavior
