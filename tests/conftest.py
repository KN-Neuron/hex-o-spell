import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification


@pytest.fixture
def sample_data():
    """Generate sample EEG-like data for testing"""
    X, y = make_classification(
        n_samples=100,
        n_features=64,
        n_classes=4,
        n_informative=40,
        n_redundant=20,
        random_state=42,
    )
    return X.astype(np.float32), y


@pytest.fixture
def sample_torch_data():
    """Generate sample torch tensors for testing"""
    X = torch.randn(100, 64, dtype=torch.float32)
    y = torch.randint(0, 4, (100,))
    return X, y


@pytest.fixture
def sample_eeg_data():
    """Generate sample EEG data with realistic parameters"""
    n_channels, n_timepoints, n_samples = 64, 1000, 50
    data = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    labels = np.random.randint(0, 4, n_samples)
    return data, labels
