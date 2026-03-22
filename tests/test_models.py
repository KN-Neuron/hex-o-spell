import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification

from src.models import EEGClassifier, train_model


def test_eeg_classifier_initialization():
    """Test EEGClassifier initialization with different parameters"""

    model = EEGClassifier(input_size=64, num_classes=4)

    assert hasattr(model, "fc1")
    assert hasattr(model, "fc2")
    assert hasattr(model, "fc3")
    assert hasattr(model, "relu")
    assert hasattr(model, "dropout")

    assert model.fc1.in_features == 64
    assert model.fc1.out_features == 128
    assert model.fc3.out_features == 4


def test_eeg_classifier_forward_pass():
    """Test EEGClassifier forward pass"""
    model = EEGClassifier(input_size=64, num_classes=4)
    dummy_input = torch.randn(32, 64, dtype=torch.float32)

    output = model(dummy_input)

    assert output.shape == (32, 4)

    assert isinstance(output, torch.Tensor)


def test_eeg_classifier_with_different_sizes():
    """Test EEGClassifier with different input sizes"""
    input_sizes = [16, 32, 64, 128]
    num_classes_list = [2, 3, 5, 10]

    for input_size in input_sizes:
        for num_classes in num_classes_list:
            model = EEGClassifier(input_size=input_size, num_classes=num_classes)
            dummy_input = torch.randn(16, input_size, dtype=torch.float32)
            output = model(dummy_input)

            assert output.shape == (16, num_classes)


def test_train_model_basic():
    """Test the train_model function with basic parameters"""

    X, y = make_classification(
        n_samples=100, n_features=64, n_classes=4, n_informative=40, random_state=42
    )
    X = X.astype(np.float32)

    model = EEGClassifier(input_size=64, num_classes=4)

    trained_model = train_model(model, X, y, epochs=5, lr=0.001)

    assert trained_model is not None

    assert isinstance(trained_model, EEGClassifier)


def test_train_model_output_shape_consistency():
    """Test that train_model preserves output shape expectations"""
    X, y = make_classification(
        n_samples=50,
        n_features=32,
        n_classes=3,
        n_informative=30,
        n_redundant=2,
        random_state=42,
    )
    X = X.astype(np.float32)

    model = EEGClassifier(input_size=32, num_classes=3)

    trained_model = train_model(model, X, y, epochs=3)

    test_input = torch.randn(10, 32, dtype=torch.float32)
    output = trained_model(test_input)

    assert output.shape == (10, 3)
    assert isinstance(output, torch.Tensor)


def test_model_gradients_update():
    """Test that model parameters are updated during training"""

    X = np.random.randn(20, 16).astype(np.float32)
    y = np.random.randint(0, 2, 20)

    model = EEGClassifier(input_size=16, num_classes=2)
    initial_weight = model.fc1.weight.clone().detach()

    trained_model = train_model(model, X, y, epochs=5, lr=0.01)

    trained_weight = trained_model.fc1.weight.clone().detach()

    assert not torch.allclose(initial_weight, trained_weight, atol=1e-6)


def test_model_same_initialization_with_same_seed():
    """Test that models have the same initial weights with fixed seeds"""

    torch.manual_seed(42)
    model1 = EEGClassifier(input_size=64, num_classes=4)

    torch.manual_seed(42)
    model2 = EEGClassifier(input_size=64, num_classes=4)

    param_pairs = list(zip(model1.parameters(), model2.parameters()))
    for i, (p1, p2) in enumerate(param_pairs):
        assert torch.allclose(
            p1, p2
        ), f"Parameter {i} should be identical with same seed"

    model1.eval()
    model2.eval()

    torch.manual_seed(123)
    dummy_input = torch.randn(16, 64, dtype=torch.float32)

    with torch.no_grad():
        output1 = model1(dummy_input)
        output2 = model2(dummy_input)

    assert torch.allclose(
        output1, output2
    ), "Models with same initialization should produce same outputs"


if __name__ == "__main__":
    pytest.main([__file__])
