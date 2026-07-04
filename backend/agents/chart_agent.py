"""
chart_agent.py — InsightFlow Generic Chart Generation Agent

Provides generate_custom_chart() which intelligently selects the best chart
type and computes the aggregated data structure required by the frontend.

Design principles:
  - 100% generic: no dataset-specific hardcoding
  - No external LLM: all logic is local keyword/schema matching
  - Production-safe: all outputs are JSON-serializable native Python
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (shared across chart types)
# ─────────────────────────────────────────────────────────────────────────────
_PALETTE = [
    "#8b5cf6", "#3b82f6", "#10b981", "#f59e0b",
    "#ef4444", "#06b6d4", "#ec4899", "#84cc16",
    "#f97316", "#a78bfa", "#34d399", "#fbbf24",
]


# ─────────────────────────────────────────────────────────────────────────────
# Keyword → intent maps  (all lower-case)
# ─────────────────────────────────────────────────────────────────────────────
_CHART_KEYWORDS: dict[str, list[str]] = {
    "scatter":   ["scatter", "correlation", "relationship", "versus", "vs"],
    "histogram": ["histogram", "distribution", "spread", "frequency", "how many"],
    "box":       ["box", "boxplot", "box plot", "outlier", "quartile", "iqr"],
    "pie":       ["pie", "breakdown", "proportion", "percentage", "share"],
    "heatmap":   ["heatmap", "heat map", "matrix", "co-occurrence", "cross"],
    "line":      ["line", "trend", "over time", "monthly", "yearly", "daily", "timeline", "time series"],
    "area":      ["area", "cumulative", "stacked"],
    "bar":       ["bar", "compare", "comparison", "count", "frequency by", "chart"],
}

_AGG_KEYWORDS: dict[str, list[str]] = {
    "mean":   ["mean", "average", "avg"],
    "median": ["median"],
    "sum":    ["sum", "total", "revenue", "sales"],
    "count":  ["count", "number of", "how many", "frequency"],
}

# Generic column-level synonym groups (std_name → synonyms)
_COL_SYNONYMS: dict[str, list[str]] = {
    "sex":      ["gender", "sex"],
    "survived": ["survival", "survive", "survived", "alive", "death", "died"],
    "pclass":   ["class", "passenger class", "ticket class"],
    "revenue":  ["revenue", "sales", "turnover", "income"],
    "profit":   ["profit", "gain", "earnings"],
    "cost":     ["cost", "expense", "expenses", "spend"],
    "quantity": ["quantity", "qty", "units", "items"],
    "date":     ["date", "time", "timestamp", "created", "order date"],
    "age":      ["age", "years old"],
    "fare":     ["fare", "ticket fare", "price", "ticket price"],
    "salary":   ["salary", "wage", "pay", "compensation"],
    "score":    ["score", "rating", "grade"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_numeric(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])


def _is_categorical(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return True
    return df[col].nunique() <= 12


def _is_numerical(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    return pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 12


def _is_datetime(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_datetime64_any_dtype(df[col])


def _to_native(value: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-serializable native Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _round(value: float, n: int = 3) -> float:
    return round(float(value), n)


# ─────────────────────────────────────────────────────────────────────────────
# NL Query parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_query(
    query: str,
    df: pd.DataFrame,
    default_x: str | None = None,
    default_chart: str = "auto",
    default_agg: str = "count",
) -> tuple[str, str | None, str, str]:
    """
    Parse a natural-language query and return:
        (x_axis, y_axis, chart_type, aggregation)

    All detection is done against the *original* query string to avoid
    stripping artifacts when column names overlap with keywords.
    """
    original = query.lower().strip()
    mutable = original  # used only for column matching (destructive)

    cols_lower_map: dict[str, str] = {c.lower(): c for c in df.columns}

    # ── Step 1: Detect aggregation & chart type from ORIGINAL query ───────────
    agg_detected = default_agg
    for agg, kws in _AGG_KEYWORDS.items():
        if any(kw in original for kw in kws):
            agg_detected = agg
            break

    chart_detected = default_chart
    for ctype, kws in _CHART_KEYWORDS.items():
        if any(kw in original for kw in kws):
            chart_detected = ctype
            break

    # ── Step 2: Column matching (greedy, longest-match first) ────────────────
    # Try exact column name first
    matched: list[str] = []
    for col_lower, col_orig in sorted(cols_lower_map.items(), key=lambda x: -len(x[0])):
        if col_lower in mutable:
            matched.append(col_orig)
            mutable = mutable.replace(col_lower, " ", 1)

    # Synonym fallback for unmatched columns
    for col_lower, col_orig in cols_lower_map.items():
        if col_orig in matched:
            continue
        for std_col, syns in _COL_SYNONYMS.items():
            if std_col == col_lower:
                for syn in sorted(syns, key=len, reverse=True):
                    if syn in original and col_orig not in matched:
                        matched.append(col_orig)
                        break

    # ── Step 3: Resolve x / y ────────────────────────────────────────────────
    fallback_x = default_x or df.columns[0]
    x_axis = matched[0] if matched else fallback_x
    y_axis: str | None = matched[1] if len(matched) > 1 else None

    # Prefer categorical on X, numeric on Y for comparison charts
    if y_axis and _is_numerical(df, x_axis) and _is_categorical(df, y_axis):
        x_axis, y_axis = y_axis, x_axis

    return x_axis, y_axis, chart_detected, agg_detected


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helper
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(df: pd.DataFrame, group_col: str, value_col: str, agg: str) -> pd.Series:
    agg_fn_map = {
        "mean":   "mean",
        "median": "median",
        "sum":    "sum",
        "count":  "count",
    }
    fn = agg_fn_map.get(agg.lower(), "count")
    return df.groupby(group_col)[value_col].agg(fn)


# ─────────────────────────────────────────────────────────────────────────────
# Per-chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_histogram(df: pd.DataFrame, x_col: str) -> tuple[str, list[dict]]:
    series = df[x_col].dropna()
    if series.empty:
        return f"Distribution of {x_col}", []

    # Sturges' rule for bin count, clamped to a reasonable range
    n = len(series)
    n_bins = max(5, min(50, int(1 + 3.322 * np.log10(n))))

    # Detect if the column is integer-like
    is_int_like = pd.api.types.is_integer_dtype(series) or np.allclose(series, series.round(0))

    if is_int_like:
        # Use clean integer bin edges
        s_min = int(series.min())
        s_max = int(series.max())
        step = max(1, round((s_max - s_min) / n_bins))
        # Round step to a "nice" number
        magnitude = 10 ** (len(str(step)) - 1)
        step = max(1, round(step / magnitude) * magnitude)
        edges = list(range(s_min, s_max + step + 1, step))
        counts, bin_edges = np.histogram(series, bins=edges)
        fmt = lambda v: str(int(v))
    else:
        counts, bin_edges = np.histogram(series, bins=n_bins)
        fmt = lambda v: f"{v:.2f}"

    data = [
        {
            "bin": f"{fmt(bin_edges[i])} – {fmt(bin_edges[i+1])}",
            "count": int(counts[i]),
        }
        for i in range(len(counts))
        if counts[i] > 0  # skip empty bins for cleaner visualisation
    ]
    return f"Distribution of {x_col}", data


def _build_box(df: pd.DataFrame, x_col: str, y_col: str | None, x_is_num: bool, y_is_num: bool) -> tuple[str, list[dict]]:
    if y_col:
        cat_c = y_col if x_is_num else x_col
        num_c = x_col if x_is_num else y_col
        groups = df.groupby(cat_c)[num_c]
        data = []
        for name, g in list(groups)[:12]:
            g_clean = g.dropna()
            if g_clean.empty:
                continue
            data.append({
                "category":  str(name),
                "min":    _round(g_clean.min()),
                "q1":     _round(g_clean.quantile(0.25)),
                "median": _round(g_clean.median()),
                "q3":     _round(g_clean.quantile(0.75)),
                "max":    _round(g_clean.max()),
            })
        title = f"{num_c} Spread by {cat_c}"
    else:
        series = df[x_col].dropna()
        data = [{
            "category":  x_col,
            "min":    _round(series.min()),
            "q1":     _round(series.quantile(0.25)),
            "median": _round(series.median()),
            "q3":     _round(series.quantile(0.75)),
            "max":    _round(series.max()),
        }] if not series.empty else []
        title = f"Spread of {x_col}"
    return title, data


def _build_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[str, list[dict]]:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) > 200:
        sub = sub.sample(200, random_state=42)
    data = [
        {"x": _round(row[x_col]), "y": _round(row[y_col])}
        for _, row in sub.iterrows()
    ]
    return f"Scatter: {y_col} vs {x_col}", data


def _build_pie(df: pd.DataFrame, x_col: str) -> tuple[str, list[dict]]:
    counts = df[x_col].value_counts().head(8)
    total = float(counts.sum())
    data = [
        {
            "name":  str(k),
            "value": _round(float(v) / total * 100.0, 1),
            "color": _PALETTE[i % len(_PALETTE)],
        }
        for i, (k, v) in enumerate(counts.items())
    ]
    return f"Breakdown of {x_col}", data


def _build_heatmap(df: pd.DataFrame, x_col: str, y_col: str | None) -> tuple[str, dict]:
    if y_col:
        ct = pd.crosstab(df[x_col], df[y_col]).head(8)
        x_labels = [str(c) for c in ct.columns]
        y_labels = [str(r) for r in ct.index]
        matrix = ct.values.tolist()
        title = f"Heatmap: {x_col} vs {y_col}"
    else:
        num_cols = df.select_dtypes(include="number").columns.tolist()[:6]
        if len(num_cols) < 2:
            return "Correlation Heatmap", {"xLabels": [], "yLabels": [], "matrix": []}
        corr = df[num_cols].corr()
        x_labels = num_cols
        y_labels = num_cols
        matrix = [
            [_round(float(corr.loc[r, c])) if not np.isnan(corr.loc[r, c]) else 0 for c in num_cols]
            for r in num_cols
        ]
        title = "Correlation Heatmap"
    return title, {"xLabels": x_labels, "yLabels": y_labels, "matrix": matrix}


def _build_line_area(df: pd.DataFrame, x_col: str, y_col: str, agg: str) -> tuple[str, list[dict]]:
    temp = df[[x_col, y_col]].dropna()
    grouped = _aggregate(temp, x_col, y_col, agg).reset_index().head(50)

    # Attempt date parsing for proper chronological ordering
    try:
        parsed = pd.to_datetime(grouped[x_col], errors="coerce")
        if not parsed.isnull().all():
            grouped[x_col] = parsed
            grouped = grouped.sort_values(x_col)
            grouped[x_col] = grouped[x_col].dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    data = [
        {"date": str(row[x_col]), "value": _round(float(row[y_col]))}
        for _, row in grouped.iterrows()
    ]
    return f"{agg.title()} {y_col} over {x_col}", data


def _build_grouped_bar(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[str, list[dict], list[str]]:
    ct = pd.crosstab(df[x_col], df[y_col]).head(10)
    series_keys = [str(c) for c in ct.columns]
    data = [
        {**{"category": str(r)}, **{str(c): int(ct.loc[r, c]) for c in ct.columns}}
        for r in ct.index
    ]
    return f"{y_col} by {x_col}", data, series_keys


def _build_bar(df: pd.DataFrame, x_col: str, y_col: str | None, agg: str) -> tuple[str, list[dict]]:
    if y_col:
        grouped = _aggregate(df[[x_col, y_col]].dropna(), x_col, y_col, agg).head(20)
        data = [{"category": str(k), "value": _round(float(v))} for k, v in grouped.items()]
        title = f"{agg.title()} {y_col} by {x_col}"
    else:
        counts = df[x_col].value_counts().head(20)
        data = [{"category": str(k), "value": int(v)} for k, v in counts.items()]
        title = f"Count of {x_col}"
    return title, data


# ─────────────────────────────────────────────────────────────────────────────
# Chart type auto-resolver
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_chart_type(
    df: pd.DataFrame,
    x_col: str,
    y_col: str | None,
    chart_type: str,
) -> str:
    if chart_type.lower() != "auto":
        return chart_type.lower().replace(" ", "-").replace("_", "-")
    return _get_recommended_chart(df, x_col, y_col)


def _get_recommended_chart(df: pd.DataFrame, x_col: str, y_col: str | None) -> str:
    x_cat = _is_categorical(df, x_col)
    x_dt  = _is_datetime(df, x_col)

    if y_col is None:
        return "bar" if x_cat else "histogram"

    y_cat = _is_categorical(df, y_col)
    y_num = _is_numerical(df, y_col)

    if x_dt and y_num:
        return "line"

    if not x_cat and not y_cat:
        return "scatter"

    if x_cat and y_cat:
        return "grouped-bar"

    # One categorical, one numerical
    return "box"


def _is_ideal_choice(df: pd.DataFrame, x_col: str, y_col: str | None, chosen: str) -> bool:
    ct = chosen.lower().replace(" ", "-").replace("_", "-")
    if ct == "auto":
        return True

    x_cat = _is_categorical(df, x_col)
    if not y_col:
        if x_cat:
            return ct in ("bar", "pie")
        else:
            return ct in ("histogram", "box")
    else:
        y_cat = _is_categorical(df, y_col)
        if x_cat and y_cat:
            return ct in ("grouped-bar", "heatmap", "bar")
        elif not x_cat and not y_cat:
            return ct in ("scatter", "line", "area", "heatmap")
        else:
            # One categorical, one numerical
            return ct in ("box", "bar", "line", "area")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_custom_chart(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str | None = None,
    chart_type: str = "auto",
    aggregation: str = "count",
    query: str = "",
) -> dict:
    """
    Generate chart data for a given DataFrame and configuration.
    """
    from agents.graph_llm_agent import decide_custom_graph_llm

    # ── Call Graph LLM Agent for smart recommendations ────────────────────
    decision = decide_custom_graph_llm(
        df=df,
        x_axis=x_axis,
        y_axis=y_axis,
        chart_type=chart_type,
        aggregation=aggregation,
        query=query
    )

    resolved_x = decision["xAxis"]
    resolved_y = decision["yAxis"]
    resolved_type = decision["chartType"]
    resolved_agg = decision["aggregation"]
    title = decision["title"]

    # ── Validate columns ──────────────────────────────────────────────────────
    available = set(df.columns)
    if resolved_x not in available:
        resolved_x = df.columns[0]
    if resolved_y and resolved_y not in available:
        resolved_y = None

    x_is_num = _is_numeric(df, resolved_x)
    y_is_num = _is_numeric(df, resolved_y) if resolved_y else False

    series_keys: list[str] | None = None
    data: Any = []

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if resolved_type == "histogram":
        _, data = _build_histogram(df, resolved_x)

    elif resolved_type in ("box", "box-plot"):
        _, data = _build_box(df, resolved_x, resolved_y, x_is_num, y_is_num)
        resolved_type = "box"

    elif resolved_type == "scatter":
        col_y = resolved_y if resolved_y else resolved_x
        _, data = _build_scatter(df, resolved_x, col_y)

    elif resolved_type == "pie":
        _, data = _build_pie(df, resolved_x)

    elif resolved_type in ("heatmap", "heat-map"):
        _, data = _build_heatmap(df, resolved_x, resolved_y)
        resolved_type = "heatmap"

    elif resolved_type in ("line", "area"):
        col_y = resolved_y if resolved_y else resolved_x
        _, data = _build_line_area(df, resolved_x, col_y, resolved_agg)

    elif resolved_type == "grouped-bar":
        if resolved_y:
            _, data, series_keys = _build_grouped_bar(df, resolved_x, resolved_y)
        else:
            _, data = _build_bar(df, resolved_x, None, resolved_agg)
            resolved_type = "bar"

    else:
        # Fallback → bar
        _, data = _build_bar(df, resolved_x, resolved_y, resolved_agg)
        resolved_type = "bar"

    recommended_type = _get_recommended_chart(df, resolved_x, resolved_y)
    is_ideal = _is_ideal_choice(df, resolved_x, resolved_y, resolved_type)

    ret = {
        "id":                   f"chart_{uuid.uuid4().hex[:8]}",
        "title":                title,
        "chartType":            resolved_type,
        "xAxis":                resolved_x,
        "yAxis":                resolved_y,
        "seriesKeys":           series_keys,
        "recommendedChartType": recommended_type,
        "isIdeal":              is_ideal,
        "xLabel":               resolved_x,
        "yLabel":               resolved_y if resolved_y else ("Frequency" if resolved_type == "histogram" else "Count")
    }

    if resolved_type == "histogram":
        ret["bins"] = data
    elif resolved_type == "heatmap":
        ret["xLabels"] = data["xLabels"]
        ret["yLabels"] = data["yLabels"]
        ret["matrix"] = data["matrix"]
    else:
        ret["data"] = data

    return ret
