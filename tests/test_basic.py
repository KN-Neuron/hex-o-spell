import numpy as np
import torch

from src.models import EEGClassifier
from src.preprocessing import EEGPreprocessor


def test_imports():
    """Test that all modules can be imported correctly"""
    try:
        from src import models, preprocessing

        # Verify that both modules have been imported successfully
        assert hasattr(models, "EEGClassifier")
        assert hasattr(preprocessing, "EEGPreprocessor")

        assert True
    except ImportError as e:
        assert False, f"Failed to import modules: {e}"


def test_preprocessor():
    """Test the EEG preprocessor"""
    preprocessor = EEGPreprocessor()
    data = np.random.rand(100, 10)
    processed_data = preprocessor.preprocess(data)

    assert processed_data.shape == data.shape

    assert np.allclose(np.mean(processed_data, axis=0), 0, atol=1e-6)
    assert np.allclose(np.std(processed_data, axis=0), 1, atol=1e-6)


def test_model():
    """Test the EEG classifier model"""
    model = EEGClassifier(input_size=10, num_classes=4)
    dummy_input = np.random.rand(5, 10).astype(np.float32)

    dummy_tensor = torch.FloatTensor(dummy_input)
    output = model(dummy_tensor)

    assert output.shape == (5, 4)


def test_model_and_preprocessor_integration():
    """Test that model and preprocessor work together"""
    # Create preprocessor and process data
    preprocessor = EEGPreprocessor()
    raw_data = np.random.rand(50, 20).astype(np.float32)
    processed_data = preprocessor.preprocess(raw_data)

    assert processed_data.shape == raw_data.shape
    assert np.allclose(np.mean(processed_data, axis=0), 0, atol=1e-6)
    assert np.allclose(np.std(processed_data, axis=0), 1, atol=1e-6)

    model = EEGClassifier(input_size=20, num_classes=3)
    tensor_data = torch.FloatTensor(processed_data[:10])
    output = model(tensor_data)

    assert output.shape == (10, 3)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
