"""
main.py — InsightFlow API v3
Orchestrates the full agent pipeline:
  Upload → clean → analyze → predict → report → /chat
"""

import io
import uuid
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.cleaning_agent import clean_data
from agents.analysis_agent import analyze_data
from agents.prediction_agent import run_prediction
from agents.report_agent import generate_report
from agents.chat_agent import get_chat_response
from agents.chart_agent import generate_custom_chart, parse_query
from agents.graph_agent import get_best_graph

app = FastAPI(title="InsightFlow API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache of the last uploaded dataset context (for /chat)
_dataset_store: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "InsightFlow API running", "version": "3.0.0"}


# ─────────────────────────────────────────────────────────────────────────────
# Upload & full pipeline
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    filename_lower = (file.filename or "").lower()
    if not any(filename_lower.endswith(ext) for ext in [".csv", ".xlsx", ".xls"]):
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Please upload a .csv or .xlsx file.",
        )

    contents = await file.read()

    # ── Parse file ────────────────────────────────────────────────
    try:
        if filename_lower.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Uploaded file is empty or has no readable data.")

    # ── Run pipeline ──────────────────────────────────────────────
    cleaning_data = clean_data(df)
    analysis = analyze_data(df)
    prediction = run_prediction(df)

    dataset_context = {
        "columns": list(df.columns),
        "rows": len(df),
        "columnInfo": {}
    }

    for col in df.columns:
        dataset_context["columnInfo"][col] = {
            "dtype": str(df[col].dtype),
            "unique": int(df[col].nunique()),
            "missing": int(df[col].isnull().sum()),
            "sampleValues": df[col].dropna().head(5).tolist()
        }

    try:
        dashboard_graph = get_best_graph(dataset_context, "Generate top 4 best dashboard graphs")
    except Exception as e:
        print("Graph generation failed:", e)
        dashboard_graph = []
    
    report = generate_report(cleaning_data, analysis, prediction)

    # ── Compute file size string ──────────────────────────────────
    size_bytes = len(contents)
    if size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"

    # ── Build response matching unified contract ──────────────────
    dataset_id = (
        (file.filename or "dataset")
        .replace(".", "_")
        .replace(" ", "_")
        + "_"
        + str(uuid.uuid4())[:8]
    )

    response = {
    "id": dataset_id,
    "name": file.filename,
    "size": size_str,
    "rows": len(df),
    "cols": len(df.columns),
    "headers": list(df.columns),
    "previewData": df.head(10).fillna("").to_dict(orient="records"),

    "stats": cleaning_data,
    "datasetType": analysis["datasetType"],
    "columnInfo": analysis["columnInfo"],
    "columnProfiles": analysis["columnProfiles"],
    "insights": analysis["insights"],
    "recommendations": analysis["recommendations"],
    "chartsData": analysis["chartsData"],
    "predictions": prediction,
    "report": report,
    "dashboardGraph": dashboard_graph,
    }

    # Cache for /chat endpoint
    _dataset_store["current"] = response
    _dataset_store["df"] = df

    return response


# ─────────────────────────────────────────────────────────────────────────────
# Chat endpoint
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    datasetContext: dict = {}

class GraphRequest(BaseModel):
    query: str
    datasetContext: dict = {}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    context = req.datasetContext if req.datasetContext else _dataset_store.get("current", {})
    # get_chat_response now returns {answer, source, intent, confidence}
    result = get_chat_response(req.question, context)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Custom Chart Generation Endpoint
# ─────────────────────────────────────────────────────────────────────────────
class CustomChartRequest(BaseModel):
    x_axis: str
    y_axis: str | None = None
    chart_type: str
    aggregation: str
    ai_query: str | None = None


@app.post("/custom-chart")
async def custom_chart_endpoint(req: CustomChartRequest):
    df = _dataset_store.get("df")
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No active dataset loaded in memory.")

    try:
        result = generate_custom_chart(
            df=df,
            x_axis=req.x_axis,
            y_axis=req.y_axis or None,
            chart_type=req.chart_type,
            aggregation=req.aggregation,
            query=req.ai_query or "",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to generate custom chart: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# /generate-chart  — agent-powered endpoint
# ─────────────────────────────────────────────────────────────────────────────
class GenerateChartRequest(BaseModel):
    xAxis: str
    yAxis: str | None = None
    chartType: str = "auto"
    aggregation: str = "count"
    query: str = ""


@app.post("/generate-chart")
async def generate_chart_endpoint(req: GenerateChartRequest):
    """
    Delegate chart generation to agents/chart_agent.py.

    - If `query` is non-empty, the NL parser overrides xAxis/yAxis/chartType/aggregation.
    - chart_type='auto' triggers schema-aware inference.
    - All aggregation modes (count, mean, median, sum) are supported.
    """
    df: pd.DataFrame | None = _dataset_store.get("df")
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No active dataset. Upload a file first.")

    try:
        result = generate_custom_chart(
            df=df,
            x_axis=req.xAxis,
            y_axis=req.yAxis or None,
            chart_type=req.chartType,
            aggregation=req.aggregation,
            query=req.query,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# /columns  — convenience: list available columns for the active dataset
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/columns")
async def get_columns():
    df: pd.DataFrame | None = _dataset_store.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No active dataset loaded.")
    cols = []
    for col in df.columns:
        cols.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "isNumeric": bool(pd.api.types.is_numeric_dtype(df[col])),
            "isDatetime": bool(pd.api.types.is_datetime64_any_dtype(df[col])),
            "uniqueCount": int(df[col].nunique()),
            "nullCount": int(df[col].isnull().sum()),
        })
    return {"columns": cols, "rowCount": len(df)}


# ─────────────────────────────────────────────────────────────────────────────
# Legacy stub
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/run-analysis")
async def run_analysis():
    return {
        "status": "success",
        "message": "Use /upload endpoint for full pipeline results.",
    }

@app.post("/generate-graph")
async def generate_graph(req: GraphRequest):
    context = req.datasetContext if req.datasetContext else _dataset_store.get("current", {})

    graph = get_best_graph(context, req.query)

    return {
        "graph": graph
    }