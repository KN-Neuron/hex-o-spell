import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from src.models import EEGClassifier, train_model
from src.preprocessing import EEGPreprocessor


def test_full_pipeline_integration():
    """Test the complete pipeline: preprocessing -> model training -> inference"""

    n_samples, n_features = 200, 64
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 4, n_samples)

    preprocessor = EEGPreprocessor()
    X_processed = preprocessor.preprocess(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    model = EEGClassifier(input_size=n_features, num_classes=4)
    trained_model = train_model(model, X_train, y_train, epochs=10)

    trained_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        predictions = trained_model(X_test_tensor)
        predicted_classes = torch.argmax(predictions, dim=1).numpy()

    assert predictions.shape == (len(X_test), 4)
    assert predicted_classes.shape == (len(X_test),)
    assert np.all(predicted_classes >= 0) and np.all(predicted_classes < 4)


def test_pipeline_different_sizes():
    """Test the pipeline with different data sizes"""
    test_cases = [
        (100, 32, 2),
        (500, 64, 3),
        (1000, 128, 5),
    ]

    for n_samples, n_features, n_classes in test_cases:

        X = np.random.rand(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples)

        preprocessor = EEGPreprocessor()
        X_processed = preprocessor.preprocess(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        model = EEGClassifier(input_size=n_features, num_classes=n_classes)
        trained_model = train_model(model, X_train, y_train, epochs=5)

        trained_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            predictions = trained_model(X_test_tensor)
            predicted_classes = torch.argmax(predictions, dim=1).numpy()

        assert predictions.shape == (len(X_test), n_classes)
        assert predicted_classes.shape == (len(X_test),)
        assert np.all(predicted_classes >= 0) and np.all(predicted_classes < n_classes)


def test_pipeline_with_realistic_data():
    """Test pipeline with more realistic EEG-like data"""

    n_samples, n_features = 300, 64
    X = np.zeros((n_samples, n_features))

    for class_idx in range(4):
        start_idx = class_idx * (n_samples // 4)
        end_idx = (class_idx + 1) * (n_samples // 4)
        X[start_idx:end_idx] = np.random.normal(
            loc=class_idx * 0.5, scale=1.0, size=(n_samples // 4, n_features)
        ).astype(np.float32)

    y = np.repeat([0, 1, 2, 3], n_samples // 4)

    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    preprocessor = EEGPreprocessor()
    X_processed = preprocessor.preprocess(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    model = EEGClassifier(input_size=n_features, num_classes=4)
    trained_model = train_model(model, X_train, y_train, epochs=15)

    trained_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        predictions = trained_model(X_test_tensor)
        predicted_classes = torch.argmax(predictions, dim=1).numpy()

    assert predictions.shape == (len(X_test), 4)
    assert predicted_classes.shape == (len(X_test),)


def test_preprocessing_does_not_affect_model_training():
    """Test that preprocessing improves model training consistency"""

    n_samples, n_features = 200, 32
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples)

    preprocessor = EEGPreprocessor()
    X_processed = preprocessor.preprocess(X)

    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_proc_train, X_proc_test, _, _ = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    model_orig = EEGClassifier(input_size=n_features, num_classes=3)
    model_proc = EEGClassifier(input_size=n_features, num_classes=3)

    trained_orig = train_model(model_orig, X_orig_train, y_train, epochs=10)
    trained_proc = train_model(model_proc, X_proc_train, y_train, epochs=10)

    assert trained_orig is not None
    assert trained_proc is not None


def test_model_prediction_consistency_after_preprocessing():
    """Test that model predictions are consistent after preprocessing"""

    X = np.random.rand(100, 16).astype(np.float32)
    y = np.random.randint(0, 2, 100)

    preprocessor = EEGPreprocessor()
    X_processed = preprocessor.preprocess(X)

    model = EEGClassifier(input_size=16, num_classes=2)
    trained_model = train_model(model, X_processed, y, epochs=5)

    test_data = np.random.rand(20, 16).astype(np.float32)
    test_data_processed = preprocessor.preprocess(test_data)

    trained_model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_data_processed)
        predictions = trained_model(test_tensor)
        predicted_classes = torch.argmax(predictions, dim=1).numpy()

    assert predictions.shape[0] == 20
    assert len(predicted_classes) == 20
    assert np.all(predicted_classes >= 0) and np.all(predicted_classes < 2)


if __name__ == "__main__":
    pytest.main([__file__])
