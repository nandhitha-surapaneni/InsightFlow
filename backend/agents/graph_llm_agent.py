from __future__ import annotations
import os
import json
import logging
import pandas as pd
from groq import Groq

logger = logging.getLogger(__name__)

# Initialize Groq client
_api_key = os.getenv("GROQ_API_KEY")
_client = Groq(api_key=_api_key) if _api_key else None

def get_columns_metadata(df: pd.DataFrame) -> list[dict]:
    metadata = []
    for col in df.columns:
        col_type = "categorical"
        is_num = pd.api.types.is_numeric_dtype(df[col])
        is_dt = pd.api.types.is_datetime64_any_dtype(df[col])
        
        # Check if datetime by looking at column name or trial parse
        if not is_dt:
            col_lower = str(col).lower()
            if any(k in col_lower for k in ["date", "time", "year", "month", "day", "timestamp", "created"]):
                try:
                    sample = df[col].dropna().head(5)
                    if not sample.empty:
                        pd.to_datetime(sample)
                        is_dt = True
                except:
                    pass
        
        unique_vals = df[col].dropna().unique()
        nunique = len(unique_vals)
        null_count = int(df[col].isnull().sum())
        
        is_bin = nunique == 2
        
        if is_dt:
            col_type = "datetime"
        elif is_num:
            if nunique <= 6:
                col_type = "categorical (low cardinality numeric)"
            else:
                col_type = "numerical"
        else:
            col_type = "categorical"
            
        sample_vals = [str(x) for x in unique_vals[:3]]
        
        metadata.append({
            "name": str(col),
            "type": col_type,
            "nunique": nunique,
            "null_count": null_count,
            "is_binary": is_bin,
            "sample_values": sample_vals
        })
    return metadata

def _is_categorical(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return True
    return df[col].nunique() <= 6

def _is_datetime(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return True
    col_lower = str(col).lower()
    if any(k in col_lower for k in ["date", "time", "year", "month", "day", "timestamp", "created"]):
        return True
    return False

def _is_numerical(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    return pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 6

def decide_custom_graph_fallback(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str | None = None,
    chart_type: str = "auto",
    aggregation: str = "count",
    query: str | None = None
) -> dict:
    if query:
        from agents.chart_agent import parse_query
        x_axis, y_axis, chart_type, aggregation = parse_query(
            query, df, default_x=x_axis, default_chart=chart_type, default_agg=aggregation
        )
    
    # Clean inputs
    if x_axis not in df.columns:
        x_axis = df.columns[0]
    if y_axis and y_axis not in df.columns:
        y_axis = None
        
    x_cat = _is_categorical(df, x_axis)
    x_dt = _is_datetime(df, x_axis)
    
    y_cat = _is_categorical(df, y_axis) if y_axis else False
    y_dt = _is_datetime(df, y_axis) if y_axis else False
    
    resolved_chart = chart_type.lower() if chart_type else "auto"
    
    # Semantic axis swapping if needed
    if y_axis and (not x_cat and y_cat):
        x_axis, y_axis = y_axis, x_axis
        x_cat, y_cat = y_cat, x_cat
        x_dt, y_dt = y_dt, x_dt

    if resolved_chart == "auto":
        if y_axis is None:
            resolved_chart = "bar" if x_cat else "histogram"
        else:
            if x_dt:
                resolved_chart = "line"
            elif not x_cat and not y_cat:
                resolved_chart = "scatter"
            elif x_cat and y_cat:
                resolved_chart = "grouped-bar"
            else:
                resolved_chart = "bar"
                
    # Extra validation for numerical vs binary / categorical
    if resolved_chart == "scatter" and y_axis:
        if x_cat or y_cat:
            resolved_chart = "bar"
            
    # Title generator
    if y_axis:
        title = f"{y_axis} by {x_axis}"
        if resolved_chart == "scatter":
            title = f"{y_axis} vs {x_axis}"
    else:
        title = f"Distribution of {x_axis}" if resolved_chart in ["histogram"] else f"Breakdown of {x_axis}"

    return {
        "chartType": resolved_chart,
        "xAxis": x_axis,
        "yAxis": y_axis,
        "aggregation": aggregation,
        "title": title
    }

def _enforce_diversity(charts: list[dict], df: pd.DataFrame) -> list[dict]:
    """
    Post-processing pass that enforces chart type diversity caps:
    - Max 1 scatter
    - Max 1 box
    - Max 1 heatmap
    - At least 1 bar
    - Remove exact duplicate (xAxis, yAxis, chartType) combos
    Returns up to 6 charts, ordered to put highest-clarity charts first.
    """
    CAPS = {"scatter": 1, "heatmap": 1, "pie": 1, "line": 1, "histogram": 1, "box": 0}
    type_counts: dict[str, int] = {}
    seen_pairs: set[tuple] = set()
    result = []

    for chart in charts:
        ctype = chart.get("chartType", "bar")
        x = chart.get("xAxis")
        y = chart.get("yAxis")
        pair = (ctype, x, y)

        # Drop duplicates
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        cap = CAPS.get(ctype, 99)  # bar has no cap
        current = type_counts.get(ctype, 0)
        if current >= cap:
            continue

        type_counts[ctype] = current + 1
        result.append(chart)

    return result[:6]


def recommend_dashboard_graphs_fallback(df: pd.DataFrame) -> list[dict]:
    charts = []
    numeric_cols = [c for c in df.columns if _is_numerical(df, c)]
    categorical_cols = [c for c in df.columns if _is_categorical(df, c)]
    dt_cols = [c for c in df.columns if _is_datetime(df, c)]

    ignore_keywords = ["id", "index", "serial", "key", "uuid", "code", "passengerid", "rownum"]
    numeric_cols = [c for c in numeric_cols if not any(k in str(c).lower() for k in ignore_keywords)]
    categorical_cols = [c for c in categorical_cols if not any(k in str(c).lower() for k in ignore_keywords)]

    # Prefer low-cardinality categoricals for pie (<=5 unique values)
    pie_candidates = [c for c in categorical_cols if 2 <= df[c].nunique() <= 5]
    bar_candidates = [c for c in categorical_cols if df[c].nunique() <= 15]

    # ── SLOT 1: Bar chart (guaranteed — highest business clarity) ─────────────
    if bar_candidates:
        charts.append({
            "chartType": "bar",
            "xAxis": bar_candidates[0],
            "yAxis": None,
            "aggregation": "count",
            "title": f"Distribution of {bar_candidates[0]}"
        })

    # ── SLOT 2: Pie chart (only if low-cardinality categorical exists) ────────
    # Pick a different column from bar if possible
    pie_col_candidates = [c for c in pie_candidates if c != (bar_candidates[0] if bar_candidates else None)]
    if not pie_col_candidates and len(pie_candidates) > 0:
        pie_col_candidates = pie_candidates  # reuse if only 1 candidate
    if pie_col_candidates:
        charts.append({
            "chartType": "pie",
            "xAxis": pie_col_candidates[0],
            "yAxis": None,
            "aggregation": "count",
            "title": f"Proportion Breakdown: {pie_col_candidates[0]}"
        })

    # ── SLOT 3: Line chart (only if datetime exists) ──────────────────────────
    if dt_cols and numeric_cols:
        charts.append({
            "chartType": "line",
            "xAxis": dt_cols[0],
            "yAxis": numeric_cols[0],
            "aggregation": "mean",
            "title": f"{numeric_cols[0]} Trend Over Time"
        })

    # ── SLOT 4: Histogram (first meaningful numeric column) ───────────────────
    if numeric_cols:
        charts.append({
            "chartType": "histogram",
            "xAxis": numeric_cols[0],
            "yAxis": None,
            "aggregation": "count",
            "title": f"Distribution of {numeric_cols[0]}"
        })

    # ── SLOT 5: Grouped bar (numeric average by category) ────────────────────────
    if numeric_cols and categorical_cols:
        # Use a different numeric col from histogram if possible
        gb_num = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        gb_cat = categorical_cols[0]
        charts.append({
            "chartType": "bar",
            "xAxis": gb_cat,
            "yAxis": gb_num,
            "aggregation": "mean",
            "title": f"Average {gb_num} by {gb_cat}"
        })

    # ── SLOT 6: Heatmap (correlation, only if 2+ numeric cols) ───────────────
    if len(numeric_cols) >= 2:
        charts.append({
            "chartType": "heatmap",
            "xAxis": numeric_cols[0],
            "yAxis": numeric_cols[1],
            "aggregation": "none",
            "title": "Correlation Heatmap"
        })

    # ── SLOT 7: Scatter (numeric vs numeric, last priority) ───────────────────
    if len(numeric_cols) >= 2:
        # Use indices 0 and 2 (or 1) to avoid reusing heatmap pair
        sx = numeric_cols[0]
        sy = numeric_cols[2] if len(numeric_cols) > 2 else numeric_cols[1]
        charts.append({
            "chartType": "scatter",
            "xAxis": sx,
            "yAxis": sy,
            "aggregation": "none",
            "title": f"{sy} vs {sx}"
        })

    # ── SLOT 8: Numeric vs Categorical bar (grouped mean) ────────────────────
    if numeric_cols and categorical_cols and len(charts) < 8:
        bc2_num = numeric_cols[0]
        bc2_cat = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0]
        charts.append({
            "chartType": "bar",
            "xAxis": bc2_cat,
            "yAxis": bc2_num,
            "aggregation": "mean",
            "title": f"Average {bc2_num} by {bc2_cat}"
        })

    return _enforce_diversity(charts, df)

def decide_custom_graph_llm(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str | None = None,
    chart_type: str = "auto",
    aggregation: str = "count",
    query: str | None = None
) -> dict:
    """
    Use Groq LLM to make intelligent, mathematically sound decisions for a single custom chart.
    """
    if not _client:
        logger.warning("Groq client not configured, falling back to rule-based logic.")
        return decide_custom_graph_fallback(df, x_axis, y_axis, chart_type, aggregation, query)
        
    metadata = get_columns_metadata(df)
    metadata_str = json.dumps(metadata, indent=2)
    
    if query:
        q = query.lower().strip()
    else:
        q = ""  

    if any(word in q for word in [" vs ", " versus ", "compare", "comparison"]):
        chart_type = "auto"

    if any(word in q for word in ["distribution", "spread", "histogram", "breakdown"]):
        chart_type = "histogram"

    if any(word in q for word in ["trend", "over time", "monthly", "yearly", "growth"]):
        chart_type = "line"

    if any(word in q for word in ["correlation", "relationship"]):
        chart_type = "scatter"
    
    if any(word in q for word in ["rate", "survival", "churn", "cancel", "conversion"]):
        if "vs" not in q:
            chart_type = "bar" 

    prompt = f"""
You are an expert data visualization assistant. Your goal is to choose the most mathematically correct and visually standard chart type, x-axis, y-axis, and aggregation for a user request.

User natural language queries may be short and ambiguous.

Examples:
- survived vs age
- fare distribution
- sales over time
- correlation between age and fare

Infer the most statistically correct graph.
Prioritize query intent heavily.

Dataset Columns Metadata:
{metadata_str}

User request parameters:
- Requested X-axis: {x_axis}
- Requested Y-axis: {y_axis}
- Requested Chart Type: {chart_type}
- Requested Aggregation: {aggregation}
- User Natural Language Query / Intent: {query or "None"}

Rules:
1. Binary columns (e.g., yes/no, survived, male/female) or low cardinality columns (unique values <= 6) should be treated as categorical/binary.
2. Numerical columns with high cardinality (unique values > 6) should be treated as numerical.
3. Datetime columns should be treated as datetime.
4. When comparing Numerical vs Numerical, choose "scatter" or "line" (trends).
5. When comparing Numerical vs Binary or Numerical vs Categorical, choose "box", "histogram", or "grouped-bar" (NEVER choose "scatter").
6. When comparing Categorical vs Categorical, choose "grouped-bar" or "heatmap".
7. Single numerical: choose "histogram".
8. Single categorical: choose "bar" or "pie".
9. Time series (datetime vs numerical): choose "line" or "area".
10. Ensure the yAxis is null if the chart type only requires one axis (like histogram, pie, or single column bar).

Output must be a valid JSON object with the following keys:
{{
  "chartType": "bar" | "line" | "scatter" | "pie" | "histogram" | "heatmap" | "area" | "grouped-bar",
  "xAxis": "column_name",
  "yAxis": "column_name" | null,
  "aggregation": "count" | "mean" | "median" | "sum" | "none",
  "title": "Descriptive Title"
}}

Return ONLY the raw JSON object. Do not include any explanations or markdown wrappers like ```json.
"""
    try:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional data visualization expert. Respond with raw JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=250
        )
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        decision = json.loads(content)
        decision["chartType"] = decision["chartType"].lower()
        if decision["xAxis"] not in df.columns:
            decision["xAxis"] = x_axis
        if decision["yAxis"] and decision["yAxis"] not in df.columns:
            decision["yAxis"] = None
        
        x_col = decision["xAxis"]
        y_col = decision["yAxis"]
        chosen = decision["chartType"]

        if not y_col:
            x_cat = _is_categorical(df, x_col)

            if x_cat and chosen not in ["bar", "pie"]:
                decision["chartType"] = "bar"
            elif not x_cat and chosen not in ["histogram", "box"]:
                decision["chartType"] = "histogram"

        else:
            x_cat = _is_categorical(df, x_col)
            y_cat = _is_categorical(df, y_col)

            if x_cat and y_cat and chosen not in ["grouped-bar", "heatmap"]:
                decision["chartType"] = "grouped-bar"

            elif (x_cat and not y_cat) or (not x_cat and y_cat):
                if "distribution" in q or "spread" in q:
                    decision["chartType"] = "histogram"
                elif chosen not in ["bar", "histogram", "grouped-bar"]:
                    decision["chartType"] = "bar"

            elif not x_cat and not y_cat:
                if chosen not in ["scatter", "line"]:
                    decision["chartType"] = "scatter"

        return decision
    except Exception as e:
        logger.error(f"LLM graph decision failed: {e}. Falling back to rule-based.")
        return decide_custom_graph_fallback(df, x_axis, y_axis, chart_type, aggregation, query)

def recommend_dashboard_graphs_llm(df: pd.DataFrame) -> list[dict]:
    """
    Use Groq LLM to suggest 4 to 8 high-value, non-redundant, and diverse dashboard charts.
    """
    if not _client:
        logger.warning("Groq client not configured, falling back to rule-based dashboard charts.")
        return recommend_dashboard_graphs_fallback(df)
        
    metadata = get_columns_metadata(df)
    metadata_str = json.dumps(metadata, indent=2)
    
    prompt = f"""
You are a senior data visualization architect. Your goal is to select a DIVERSE set of dashboard charts for a business analytics dashboard.

Dataset Columns Metadata:
{metadata_str}

DIVERSITY RULES (strictly enforced):
- MAX 1 scatter chart total (ONLY if highly meaningful)
- FORBIDDEN: box plots (DO NOT generate box plots)
- MAX 1 heatmap total (ONLY if highly useful)
- MAX 1 pie chart total (ONLY if categorical column has 5 or fewer unique values)
- MINIMUM 1 bar chart (required)
- Include a line chart ONLY if a datetime or time-like column exists
- Do NOT output multiple charts showing the same column pair
- Better to show 4 great charts than 6 bad charts. Do NOT force all 6 if dataset does not support them.

SLOT PRIORITY ORDER (fill these in order, skip if data type unavailable):
1. bar     — categorical distribution (count), most important for business overview
2. pie     — proportion breakdown (only if low-cardinality categorical ≤5 values exists)
3. histogram — numerical distribution (single numeric column)
4. line    — trend over time (only if datetime column exists)
5. scatter — two numeric variables (last resort, use different columns from heatmap)
6. heatmap — correlation matrix (requires ≥2 numeric columns)

HARD TYPE RULES:
- Categorical + Numerical → bar (NEVER scatter or box)
- Binary + Numerical → bar only
- Binary + Binary → grouped-bar or heatmap only
- Numerical + Numerical → scatter or heatmap
- Time + Numerical → line or area
- Single Categorical → bar or pie
- Single Numerical → histogram
- Do NOT select ID-like columns (id, index, key, serial, passengerid, rownum)
- Skip columns with >40% missing values

Return between 4 and 6 charts. Rank from highest business insight clarity to lowest.

Output must be a valid JSON array:
[
  {{
    "chartType": "bar" | "line" | "scatter" | "pie" | "histogram" | "heatmap" | "area" | "grouped-bar",
    "xAxis": "column_name",
    "yAxis": "column_name" | null,
    "aggregation": "count" | "mean" | "median" | "sum" | "none",
    "title": "Concise, business-readable title"
  }}
]

Return ONLY the raw JSON array. No explanations, no markdown wrappers.
"""
    try:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional dashboard designer. Respond with raw JSON array only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        charts = json.loads(content)
        if not isinstance(charts, list):
            raise ValueError("LLM response is not a JSON list")
            
        validated_charts = []

        for chart in charts:
            if chart.get("xAxis") in df.columns:
                chart["chartType"] = chart["chartType"].lower()

                if chart.get("yAxis") and chart.get("yAxis") not in df.columns:
                    chart["yAxis"] = None

                x_col = chart["xAxis"]
                y_col = chart["yAxis"]
                chosen = chart["chartType"]

                if not y_col:
                    x_cat = _is_categorical(df, x_col)

                    if x_cat and chosen not in ["bar", "pie"]:
                        chart["chartType"] = "bar"
                    elif not x_cat and chosen not in ["histogram", "box"]:
                        chart["chartType"] = "histogram"

                else:
                    x_cat = _is_categorical(df, x_col)
                    y_cat = _is_categorical(df, y_col)

                    if x_cat and y_cat and chosen not in ["grouped-bar", "heatmap"]:
                        chart["chartType"] = "grouped-bar"

                    if (x_cat and not y_cat) or (not x_cat and y_cat):
                        if chosen not in ["bar", "grouped-bar"]:
                            chart["chartType"] = "bar"

                    elif not x_cat and not y_cat:
                        if chosen not in ["scatter", "line", "heatmap"]:
                            chart["chartType"] = "scatter"

                validated_charts.append(chart)

        return _enforce_diversity(validated_charts, df)
    except Exception as e:
        logger.error(f"LLM dashboard recommendation failed: {e}. Falling back to rule-based.")
        return recommend_dashboard_graphs_fallback(df)
