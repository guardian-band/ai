from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import PredictionRequest


def test_request_normalizes_drug_ids() -> None:
    request = PredictionRequest(drug_a=" db00313 ", drug_b="db01041", top_k=5)
    assert request.drug_a == "DB00313"
    assert request.drug_b == "DB01041"


def test_request_rejects_invalid_top_k() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(drug_a="DB00313", drug_b="DB01041", top_k=0)
