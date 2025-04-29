"""Module for the FastAPI configuration."""

import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for the ML model."""

    path: Path
    version: str
    species: List[str]


class ServerConfig(BaseModel):
    """Configuration for the FastAPI server."""

    host: str
    port: int


class AppConfig(BaseModel):
    """Main application configuration."""

    model: ModelConfig
    server: ServerConfig
    version: str


def load_config() -> AppConfig:
    """Load the application configuration from a YAML file."""
    config_path = os.getenv("APP_CONFIG_PATH", "src/serve/config.yaml")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
