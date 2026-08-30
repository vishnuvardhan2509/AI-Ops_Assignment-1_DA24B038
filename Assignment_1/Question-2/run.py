import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

# --- 1. Load MNIST (subset for speed) ---
print("Loading MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X[:5000]   # Use 5000 samples for speed
y = y[:5000]

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Define hyperparameters to tune ---
learning_rates = [0.001, 0.0001]          # 2 values
batch_sizes = [32, 64, 128]               # 3 values

# This will give 2 x 3 = 6 experiments

# --- 3. Run experiments ---
for lr in learning_rates:
    for batch_size in batch_sizes:
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("hidden_layer_sizes", (64, 32))   # fixed for simplicity
            mlflow.log_param("activation", "relu")
            mlflow.log_param("solver", "adam")
            mlflow.log_param("max_iter", 100)

            # Create model
            model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                learning_rate_init=lr,
                batch_size=batch_size,
                max_iter=100,
                random_state=42,
                verbose=False
            )

            # Train
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            # Log metrics
            acc = accuracy_score(y_test, y_pred)
            loss = log_loss(y_test, y_prob)

            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_loss", loss)

            print(f"lr={lr}, batch={batch_size}, accuracy={acc:.4f}, loss={loss:.4f}")

print("All experiments done!")