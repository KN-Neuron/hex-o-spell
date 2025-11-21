import numpy as np  # noqa: F401
from sklearn.preprocessing import StandardScaler  # type: ignore
from typing import Union, Any
import numpy.typing as npt


class EEGPreprocessor:
    """Class for preprocessing EEG data"""

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def preprocess(self, data: Union[npt.ArrayLike, list]) -> Any:
        """Apply preprocessing steps to EEG data"""
        standardized_data = self.scaler.fit_transform(data)
        return standardized_data
