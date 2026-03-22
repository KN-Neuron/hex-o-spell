import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from typing import Tuple
import numpy.typing as npt

from src.models import EEGClassifier, train_model
from src.preprocessing import EEGPreprocessor


def load_sample_data() -> Tuple[npt.NDArray, npt.NDArray]:
    """Load or generate sample EEG data for demonstration"""
    n_samples, n_features = 1000, 64
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 4, n_samples)
    return X, y


def main() -> None:
    print("Loading EEG data...")
    X, y = load_sample_data()

    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

    preprocessor = EEGPreprocessor()
    X_processed = preprocessor.preprocess(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    input_size = X_train.shape[1]
    num_classes = len(np.unique(y))
    model = EEGClassifier(input_size, num_classes)

    print(f"Training model with input size {input_size} and {num_classes} classes...")

    trained_model = train_model(model, X_train, y_train, epochs=50)

    trained_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        predictions = trained_model(X_test_tensor)
        predicted_classes = torch.argmax(predictions, dim=1).numpy()

    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_classes))


if __name__ == "__main__":
    main()
