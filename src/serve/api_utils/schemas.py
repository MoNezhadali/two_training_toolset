"""
This module contains the schemas for the API requests and responses.
It defines the request and response models using Pydantic.
"""

from typing import Literal

from pydantic import BaseModel, Field


class IrisRequest(BaseModel):
    """Request model for predicting Iris species."""

    request_id: str
    sepal_length: float = Field(..., ge=0)
    sepal_width: float = Field(..., ge=0)
    petal_length: float = Field(..., ge=0)
    petal_width: float = Field(..., ge=0)


class IrisResponse(BaseModel):
    """Response model for Iris species prediction."""

    prediction: int
    prediction_label: Literal["setosa", "versicolor", "virginica"]
    request_id: str
    model_version: str
    api_version: str
