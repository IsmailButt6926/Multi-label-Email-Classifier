from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseModel(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and multi-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        ...

    @abstractmethod
    def predict(self) -> int:
        """
        Predict the output using trained model.
        """
        ...

    @abstractmethod
    def print_results(self) -> None:
        """
        Print classification report and evaluation metrics.
        """
        ...

    @abstractmethod
    def data_transform(self) -> None:
        return
