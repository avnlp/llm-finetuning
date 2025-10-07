"""Configuration models for BioASQ inference pipeline.

This module defines Pydantic models for loading and validating
the YAML configuration file used in the BioASQ inference pipeline.
"""

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for the model parameters."""

    name: str
    max_seq_length: int
    lora_rank: int
    gpu_memory_utilization: float


class GenerationConfig(BaseModel):
    """Configuration for the generation parameters."""

    temperature: float
    top_k: int
    top_p: float
    min_p: float
    max_new_tokens: int


class DatasetConfig(BaseModel):
    """Configuration for the dataset parameters."""

    name: str
    split: str


class OutputConfig(BaseModel):
    """Configuration for the output parameters."""

    file_name: str


class BioASQInferenceConfig(BaseModel):
    """Main configuration model for the BioASQ inference pipeline."""

    model: ModelConfig
    generation: GenerationConfig
    dataset: DatasetConfig
    output: OutputConfig
