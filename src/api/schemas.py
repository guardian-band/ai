from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drug_a: str = Field(min_length=1, examples=["DB00313"])
    drug_b: str = Field(min_length=1, examples=["DB01041"])
    top_k: int = Field(default=5, ge=1, le=100)

    @field_validator("drug_a", "drug_b")
    @classmethod
    def normalize_drug_id(cls, value: str) -> str:
        return value.strip().upper()


class SpecificPrediction(BaseModel):
    label: str
    cui: str
    probability: float


class OrganPrediction(BaseModel):
    organ: str
    probability: float


class PredictionResponse(BaseModel):
    pair_id: str
    drug_a: str
    drug_b: str
    model: str
    mode: Literal["fused", "baseline"]
    confidence: Literal["standard", "reduced"]
    warnings: list[str]
    level_1_organs: list[OrganPrediction]
    level_2_side_effects: list[SpecificPrediction]
    inference_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    runtime: Literal["pytorch", "onnx"]
    device: str
    selected_mode: str
    supported_drugs: int
    checkpoint_sha256: str


class ModelInfoResponse(BaseModel):
    model: str
    runtime: Literal["pytorch", "onnx"]
    selected_mode: str
    device: str
    supported_drugs: int
    level_1_labels: list[str]
    level_2_count: int
    checkpoint_sha256: str
    validation_macro_auprc: float | None
