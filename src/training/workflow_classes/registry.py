"""Registry for workflow classes."""

# Standard library imports
from typing import Any, Dict, Type

# Registry for workflow classes.
WORKFLOW_REGISTRY: Dict[str, Type[Any]] = {}


def register_workflow(workflow_class: Type[Any]) -> Type[Any]:
    """
    Decorator to register an workflow class.

    Parameters
    ----------
    workflow_class: Type[Any]
        The workflow class to register.

    Returns
    -------
    Type[Any]:
        The workflow class.

    """
    class_name = workflow_class.__name__
    WORKFLOW_REGISTRY[class_name] = workflow_class
    return workflow_class


def get_workflow_class(name: str) -> Type[Any]:
    """
    Retrieve a workflow class by its name.

    Parameters
    ----------
    name: str
        The name of the workflow class.

    Returns
    -------
    Type[Any]:
        The workflow class.

    """
    workflow_class = WORKFLOW_REGISTRY.get(name)
    if not workflow_class:
        raise ValueError(f"Unknown workflow class: {name}")
    return workflow_class
