import pandas as pd
import numpy as np


def generate_visualizations(df: pd.DataFrame):
    charts = []

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ---------- Line Chart ----------
    if numeric_cols:
        col = numeric_cols[0]
        try:
            values = df[col].dropna()
            sample = values.head(20)

            line_data = [
                {"date": str(i + 1), "value": round(float(v), 2)}
                for i, v in enumerate(sample)
            ]

            charts.append({
                "type": "line",
                "title": f"Trend Analysis - {col}",
                "data": line_data
            })
        except:
            pass

    # ---------- Bar Chart ----------
    if categorical_cols:
        col = categorical_cols[0]
        try:
            counts = df[col].value_counts().head(10)

            bar_data = [
                {"category": str(k), "value": int(v)}
                for k, v in counts.items()
            ]

            charts.append({
                "type": "bar",
                "title": f"Distribution - {col}",
                "data": bar_data
            })
        except:
            pass

    # ---------- Pie Chart ----------
    if categorical_cols:
        col = categorical_cols[0]
        palette = [
            "#8b5cf6", "#3b82f6", "#10b981",
            "#f59e0b", "#ef4444", "#06b6d4"
        ]

        try:
            counts = df[col].value_counts().head(6)
            total = counts.sum()

            pie_data = [
                {
                    "name": str(k),
                    "value": round((v / total) * 100, 1),
                    "color": palette[i % len(palette)]
                }
                for i, (k, v) in enumerate(counts.items())
            ]

            charts.append({
                "type": "pie",
                "title": f"Category Breakdown - {col}",
                "data": pie_data
            })
        except:
            pass

    # ---------- Heatmap ----------
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr()
            cols = numeric_cols[:4]

            heatmap_data = []
            for i, c1 in enumerate(cols):
                for j, c2 in enumerate(cols):
                    if i < j:
                        val = corr.loc[c1, c2]
                        if not np.isnan(val):
                            heatmap_data.append({
                                "x": c1,
                                "y": c2,
                                "val": round(float(val), 2)
                            })

            charts.append({
                "type": "heatmap",
                "title": "Correlation Matrix",
                "data": heatmap_data
            })
        except:
            pass

    return charts