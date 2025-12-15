import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, log_loss


# import joblib
#
# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(pca, "pca.pkl")
# joblib.dump(label_encoder, "label_encoder.pkl")
#
# joblib.dump(log_reg, "logistic_regression.pkl")
# joblib.dump(rf, "random_forest.pkl")

# ============================
# STEP 1: PREPROCESSING
# ============================

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

# Train/validation split (80% train, 20% val)
X_train, X_val, y_train, y_val = train_test_split(
    df_X_temp, df_y_temp,
    test_size=0.2,
    random_state=42,
    stratify=df_y_temp
)

# 2) SCALING - FIT ONLY ON TRAINING DATA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(df_X_test)

# 3) LABEL ENCODING - FIT ONLY ON TRAINING DATA
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(df_y_test)

print("Classes (activities):", label_encoder.classes_)

# 4) PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Original feature count:", X_train.shape[1])
print("PCA feature count     :", X_train_pca.shape[1])

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA – Explained Variance")
plt.grid(True)
plt.show()


# ============================
# STEP 2: MODEL TRAINING WITH OVERFITTING ANALYSIS
# ============================

def train_and_evaluate(name, model):
    print(f"\n{'=' * 70}")
    print(f"Training {name}...")
    print('=' * 70)

    # Train on PCA features
    model.fit(X_train_pca, y_train_enc)
    print("✓ Model trained successfully")

    # Training set predictions
    y_train_pred = model.predict(X_train_pca)
    y_train_proba = model.predict_proba(X_train_pca)

    # Validation set predictions
    y_val_pred = model.predict(X_val_pca)
    y_val_proba = model.predict_proba(X_val_pca)

    # Test set predictions
    y_test_pred = model.predict(X_test_pca)
    y_test_proba = model.predict_proba(X_test_pca)

    # ==========================================
    # CALCULATE METRICS FOR ALL THREE SETS
    # ==========================================

    # Training metrics
    acc_train = accuracy_score(y_train_enc, y_train_pred)
    prec_train = precision_score(y_train_enc, y_train_pred, average="macro")
    loss_train = log_loss(y_train_enc, y_train_proba)

    # Validation metrics
    acc_val = accuracy_score(y_val_enc, y_val_pred)
    prec_val = precision_score(y_val_enc, y_val_pred, average="macro")
    loss_val = log_loss(y_val_enc, y_val_proba)

    # Test metrics
    acc_test = accuracy_score(y_test_enc, y_test_pred)
    prec_test = precision_score(y_test_enc, y_test_pred, average="macro")
    loss_test = log_loss(y_test_enc, y_test_proba)

    # ==========================================
    # DISPLAY ALL RESULTS
    # ==========================================

    print(f"\n{name} TRAINING Results:")
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

    # ==========================================
    # OVERFITTING / UNDERFITTING ANALYSIS
    # ==========================================

    print(f"\n{name} OVERFITTING ANALYSIS:")
    print("-" * 70)

    # Gap between train and validation (overfitting indicator)
    acc_gap_val = acc_train - acc_val
    loss_gap_val = loss_val - loss_train

    # Gap between train and test (overfitting indicator)
    acc_gap_test = acc_train - acc_test
    loss_gap_test = loss_test - loss_train

    print(f"Train vs Validation:")
    print(f"  Accuracy gap (train - val): {acc_gap_val:+.6f} {'⚠️ OVERFITTING' if acc_gap_val > 0.05 else '✓ OK'}")
    print(f"  Loss gap (val - train)    : {loss_gap_val:+.6f} {'⚠️ OVERFITTING' if loss_gap_val > 0.05 else '✓ OK'}")

    print(f"\nTrain vs Test:")
    print(f"  Accuracy gap (train - test): {acc_gap_test:+.6f} {'⚠️ OVERFITTING' if acc_gap_test > 0.05 else '✓ OK'}")
    print(f"  Loss gap (test - train)    : {loss_gap_test:+.6f} {'⚠️ OVERFITTING' if loss_gap_test > 0.05 else '✓ OK'}")

    # Underfitting check
    if acc_train < 0.85:
        print(f"\n  ⚠️ UNDERFITTING WARNING: Training accuracy is {acc_train:.4f} (< 0.85)")
    else:
        print(f"\n  ✓ Training accuracy is good: {acc_train:.4f}")

    # ==========================================
    # LEARNING CURVE (Training vs Validation Loss)
    # ==========================================

    print(f"\nGenerating learning curve for {name}...")

    params = model.get_params()
    params['n_jobs'] = 1
    model_for_curve = model.__class__(**params)

    train_sizes, train_scores, val_scores = learning_curve(
        model_for_curve,
        X_train_pca,
        y_train_enc,
        train_sizes=np.linspace(0.3, 1.0, 4),
        cv=2,
        scoring="neg_log_loss",
        shuffle=True,
        random_state=42,
        n_jobs=1
    )

    train_loss_curve = -train_scores.mean(axis=1)
    val_loss_curve = -val_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_loss_curve, marker="o", label="Training loss", linewidth=2, markersize=8)
    plt.plot(train_sizes, val_loss_curve, marker="s", label="Validation loss", linewidth=2, markersize=8)
    plt.xlabel("Training examples")
    plt.ylabel("Log loss")
    plt.title(f"Learning Curve – {name}\n(Overfitting indicator: gap between train & validation)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"✓ Learning curve generated for {name}")


# ============================
# TRAIN AND EVALUATE MODELS
# ============================

# Logistic Regression
log_reg = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train and evaluate both models
train_and_evaluate("Logistic Regression", log_reg)
train_and_evaluate("Random Forest", rf)

print("\n" + "=" * 70)
print("All models trained and evaluated successfully!")
print("=" * 70)

print("=" * 70)
# ============================
# SAVE TRAINED MODELS & PREPROCESSING (FOR STREAMLIT)
# ============================

import joblib

joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

joblib.dump(log_reg, "logistic_regression.pkl")
joblib.dump(rf, "random_forest.pkl")

print("Models and preprocessing objects saved successfully!")

