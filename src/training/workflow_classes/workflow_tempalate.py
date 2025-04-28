"""Module for defining the abstract base class for ML training workflows."""

from abc import ABC, abstractmethod


class WorkflowTemplate(ABC):
    """Abstract base class for ML training workflows."""

    @abstractmethod
    def load_data(self):
        """Load the dataset."""
        pass

    @abstractmethod
    def split_data(self):
        """Split data into training and testing sets."""
        pass

    @abstractmethod
    def build_pipeline(self):
        """Build the preprocessing and modeling pipeline."""
        pass

    @abstractmethod
    def train_model(self):
        """Train the model."""
        pass

    @abstractmethod
    def save_model(self):
        """Save the trained model and metadata."""
        pass
