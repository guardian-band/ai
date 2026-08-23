from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .model_service import ModelService, UnsupportedDrugError
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)
from .settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = ModelService(settings or Settings.from_env())
        service.load()
        app.state.model_service = service
        yield

    app = FastAPI(
        title="GuardianBand Polypharmacy AI",
        version="1.0.0",
        description="Frozen warm-pair Morgan+MPNN Teacher inference service.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadinessResponse)
    def ready(request: Request) -> ReadinessResponse:
        service: ModelService = request.app.state.model_service
        if not service.ready:
            raise HTTPException(status_code=503, detail="model is not ready")
        return ReadinessResponse(
            status="ready",
            device=str(service.device),
            selected_mode=service.selected_mode,
            supported_drugs=len(service.drug_ids),
            checkpoint_sha256=service.checkpoint_sha256,
        )

    @app.get("/v1/model", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        service: ModelService = request.app.state.model_service
        return ModelInfoResponse(
            model="warm_morgan_mpnn_teacher",
            selected_mode=service.selected_mode,
            device=str(service.device),
            supported_drugs=len(service.drug_ids),
            level_1_labels=service.organs,
            level_2_count=len(service.labels),
            checkpoint_sha256=service.checkpoint_sha256,
            validation_macro_auprc=service.validation_macro_auprc,
        )

    @app.post("/v1/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        if payload.drug_a == payload.drug_b:
            raise HTTPException(status_code=422, detail="drug_a and drug_b must differ")
        service: ModelService = request.app.state.model_service
        try:
            return service.predict(payload.drug_a, payload.drug_b, payload.top_k)
        except UnsupportedDrugError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unsupported_drug",
                    "drug_ids": exc.drug_ids,
                    "message": "Prediction abstained: one or more drugs are unsupported.",
                },
            ) from exc

    return app


app = create_app()
