"""SyncGuard AI detection API (FastAPI).
Run:   uvicorn api:app --reload        Docs (Swagger UI): http://127.0.0.1:8000/docs"""
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
import config
import service

app = FastAPI(
    title="SyncGuard AI Detection API",
    version="1.0.0",
    description="Scores pricing and inventory sync records with a trained autoencoder and flags "
                "potentially problematic syncs, with severity, reason, likely type and suggested action.")

EXAMPLE = {"timestamp": "2026-01-20T14:00:00", "sku": "SKU007", "channel": "Amazon", "master_price": 49.99,
           "channel_price": 5.15, "master_stock": 600, "channel_stock": 200, "units_sold": 6, "sync_delay_min": 4.2}


class SyncRecord(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE})
    timestamp: datetime = Field(description="When the sync happened")
    sku: str = Field(min_length=1, description="Product code")
    channel: str = Field(min_length=1, description="Sales channel, e.g. Shopify, Amazon, eBay")
    master_price: float = Field(gt=0, description="Price in the master system")
    channel_price: float = Field(ge=0, description="Price currently shown on the channel")
    master_stock: float = Field(ge=0, description="Stock in the master system")
    channel_stock: float = Field(ge=0, description="Stock currently shown on the channel")
    units_sold: float = Field(ge=0, description="Units sold in the last period")
    sync_delay_min: float = Field(ge=0, description="Minutes since the channel was last updated")


class ScoreResult(BaseModel):
    index: int
    sku: str
    channel: str
    timestamp: str
    anomaly_score: float = Field(description="Reconstruction-based score; higher = more unusual")
    threshold: float = Field(description="Alert threshold used (after sensitivity)")
    is_anomaly: bool
    severity: str = Field(description="High / Medium / Low, or None when not flagged")
    likely_type: Optional[str] = None
    reason: Optional[str] = None
    detected_by: Optional[str] = Field(None, description="Model, Rule or Both")
    estimated_revenue_at_risk: float
    suggested_action: Optional[str] = None
    feature_scores: dict[str, float] = Field(description="How unusual each feature was (1.0 = edge of normal)")


class BatchRequest(BaseModel):
    records: list[SyncRecord] = Field(min_length=1, max_length=5000)
    sensitivity: float = Field(1.0, ge=0.25, le=3.0, description="Higher = more alerts")
    use_rules: bool = Field(True, description="Also apply the simple safety rules")
    only_flagged: bool = Field(False, description="Return only the flagged records")


class BatchResponse(BaseModel):
    count: int
    flagged: int
    results: list[ScoreResult]


def _require_model():
    if not service.ready():
        raise HTTPException(503, "Model not trained yet. Run `python run_all.py` first.")


def _load(path):
    _require_model()
    return json.load(open(path))


@app.get("/", include_in_schema=False)
def root():
    return {"service": "SyncGuard AI", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["Service"], summary="Is the service up and the model loaded?")
def health():
    return {"status": "ok" if service.ready() else "model_missing", "model_loaded": service.ready(),
            "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/model/info", tags=["Service"], summary="Model description")
def model_info():
    t, m = _load(config.THRESHOLDS_FILE), _load(config.METRICS_FILE)
    return {"model": "Autoencoder (scikit-learn MLPRegressor)", "architecture": f"{len(config.FEATURE_COLUMNS)} -> "
            f"{' -> '.join(map(str, config.HIDDEN_LAYERS))} -> {len(config.FEATURE_COLUMNS)}",
            "features": config.FEATURE_COLUMNS, "threshold_method": t["chosen_method"], "threshold": t["chosen"],
            "training_rows": m["n_train"], "rows_trimmed_in_cleaning_pass": m["trimmed_rows"],
            "rules": {"price_gap": config.RULE_PRICE_DIFF, "sync_delay_min": config.RULE_SYNC_DELAY_MIN}}


@app.get("/thresholds", tags=["Service"], summary="All threshold methods and how each performs on held-out data")
def thresholds():
    return _load(config.THRESHOLDS_FILE)


@app.get("/metrics", tags=["Service"], summary="Held-out evaluation results")
def metrics():
    return _load(config.METRICS_FILE)


@app.post("/score", response_model=ScoreResult, tags=["Detection"], summary="Score one sync record")
def score(record: SyncRecord, sensitivity: float = Query(1.0, ge=0.25, le=3.0), use_rules: bool = True):
    _require_model()
    return service.score_records([record.model_dump()], sensitivity, use_rules)[0]


@app.post("/score/batch", response_model=BatchResponse, tags=["Detection"], summary="Score many sync records")
def score_batch(req: BatchRequest):
    _require_model()
    results = service.score_records([r.model_dump() for r in req.records], req.sensitivity, req.use_rules)
    flagged = sum(r["is_anomaly"] for r in results)
    if req.only_flagged:
        results = [r for r in results if r["is_anomaly"]]
    return {"count": len(req.records), "flagged": flagged, "results": results}
