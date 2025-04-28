"""General workflow using Metaflow."""

import yaml
from metaflow import FlowSpec, Parameter, step

from src.training.workflow_classes.registry import get_workflow_class


class TrainingWorkflow(FlowSpec):
    """Generic training workflow powered by Metaflow."""

    config_path = Parameter(
        "config", help="Path to the YAML config", default="config.yaml"
    )

    @step
    def start(self):
        """Load configuration and workflow class."""
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        class_name = self.config.get("workflow_class")
        WorkflowClass = get_workflow_class(class_name)

        self.workflow = WorkflowClass(self.config)
        self.next(self.load_data)

    @step
    def load_data(self):
        """Load the dataset."""
        self.workflow.load_data()
        self.next(self.split_data)

    @step
    def split_data(self):
        """Split data into train/test."""
        self.workflow.split_data()
        self.next(self.build_pipeline)

    @step
    def build_pipeline(self):
        """Build preprocessing and model pipeline."""
        self.workflow.build_pipeline()
        self.next(self.train_model)

    @step
    def train_model(self):
        """Train the model."""
        self.workflow.train_model()
        self.next(self.save_model)

    @step
    def save_model(self):
        """Save the model and metadata."""
        self.workflow.save_model()
        self.next(self.end)

    @step
    def end(self):
        """End of the flow."""
        print("Workflow completed successfully.")


if __name__ == "__main__":
    TrainingWorkflow()
