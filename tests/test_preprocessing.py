import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.preprocessing import EEGPreprocessor


def test_eeg_preprocessor_initialization():
    """Test EEGPreprocessor initialization"""
    preprocessor = EEGPreprocessor()

    assert hasattr(preprocessor, "scaler")
    assert isinstance(preprocessor.scaler, StandardScaler)


def test_eeg_preprocessor_basic_preprocessing():
    """Test basic preprocessing functionality"""
    preprocessor = EEGPreprocessor()
    data = np.random.rand(100, 10)

    processed_data = preprocessor.preprocess(data)

    assert processed_data.shape == data.shape

    assert np.allclose(np.mean(processed_data, axis=0), 0, atol=1e-6)

    assert np.allclose(np.std(processed_data, axis=0), 1, atol=1e-6)


def test_eeg_preprocessor_with_different_shapes():
    """Test preprocessing with different data shapes"""
    shapes = [(50, 5), (100, 20), (200, 10), (1000, 64)]

    for n_samples, n_features in shapes:
        preprocessor = EEGPreprocessor()
        data = np.random.rand(n_samples, n_features)

        processed_data = preprocessor.preprocess(data)

        assert processed_data.shape == data.shape
        assert np.allclose(np.mean(processed_data, axis=0), 0, atol=1e-6)
        assert np.allclose(np.std(processed_data, axis=0), 1, atol=1e-6)


def test_eeg_preprocessor_consistency():
    """Test that preprocessing is consistent for the same data"""
    preprocessor = EEGPreprocessor()
    data = np.random.rand(50, 10)

    processed1 = preprocessor.preprocess(data)
    processed2 = preprocessor.preprocess(data.copy())

    assert np.allclose(processed1, processed2)


def test_eeg_preprocessor_with_extreme_values():
    """Test preprocessing with data containing extreme values"""
    preprocessor = EEGPreprocessor()

    data = np.random.randn(100, 10) * 1000

    processed_data = preprocessor.preprocess(data)

    assert processed_data.shape == data.shape

    assert np.allclose(np.mean(processed_data, axis=0), 0, atol=1e-6)

    assert np.allclose(np.std(processed_data, axis=0), 1, atol=1e-6)


def test_eeg_preprocessor_single_feature():
    """Test preprocessing with single feature"""
    preprocessor = EEGPreprocessor()
    data = np.random.rand(100, 1)

    processed_data = preprocessor.preprocess(data)

    assert processed_data.shape == data.shape
    assert np.allclose(np.mean(processed_data), 0, atol=1e-6)
    assert np.allclose(np.std(processed_data), 1, atol=1e-6)


def test_eeg_preprocessor_preserves_data_type():
    """Test that preprocessing preserves data type"""
    preprocessor = EEGPreprocessor()
    data = np.random.rand(50, 10).astype(np.float32)

    processed_data = preprocessor.preprocess(data)

    assert processed_data.dtype == data.dtype


def test_different_preprocessing_consistency():
    """Test that different preprocessor instances give consistent relative results"""
    data = np.random.rand(100, 10)

    preprocessor1 = EEGPreprocessor()
    preprocessor2 = EEGPreprocessor()

    processed1 = preprocessor1.preprocess(data.copy())
    processed2 = preprocessor2.preprocess(data.copy())

    assert np.allclose(np.mean(processed1, axis=0), 0, atol=1e-6)
    assert np.allclose(np.std(processed1, axis=0), 1, atol=1e-6)
    assert np.allclose(np.mean(processed2, axis=0), 0, atol=1e-6)
    assert np.allclose(np.std(processed2, axis=0), 1, atol=1e-6)


def test_preprocessing_with_various_distributions():
    """Test preprocessing with different data distributions"""
    distributions = [
        np.random.rand(100, 10),
        np.random.randn(100, 10),
        np.random.exponential(1, (100, 10)),
        np.random.beta(2, 5, (100, 10)),
    ]

    for data in distributions:
        preprocessor = EEGPreprocessor()
        processed_data = preprocessor.preprocess(data)

        assert processed_data.shape == data.shape
        assert np.allclose(np.mean(processed_data, axis=0), 0, atol=1e-6)
        assert np.allclose(np.std(processed_data, axis=0), 1, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
