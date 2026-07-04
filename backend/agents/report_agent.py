"""
report_agent.py
Generates an executive summary report from analysis + prediction + cleaning results.
The summary is narrative, pulling from actual insight findings rather than metadata.
"""

from typing import Any, Dict


def generate_report(cleaning_data: Dict, analysis_data: Dict, prediction_data: Dict) -> Dict:
    summary_parts = []

    stats = cleaning_data
    dataset_type = analysis_data.get("datasetType", "Generic")
    insights = analysis_data.get("insights", [])
    recommendations = analysis_data.get("recommendations", [])

    rows = stats.get("totalRows", 0)
    cols = stats.get("totalColumns", 0)
    missing = stats.get("missingValues", 0)
    duplicates = stats.get("duplicateRows", 0)
    quality = stats.get("qualityScore", 0)

    # --- Headline
    summary_parts.append(
        f"This {dataset_type} dataset contains {rows:,} records across {cols} variables."
    )

    # --- Data quality signal
    if quality:
        quality_label = "excellent" if quality >= 85 else "moderate" if quality >= 60 else "poor"
        summary_parts.append(
            f"Overall data quality is {quality_label} (score: {quality}/100)."
        )
    if missing > 0:
        missing_pct = round((missing / max(rows * cols, 1)) * 100, 1)
        summary_parts.append(
            f"Detected {missing:,} missing values ({missing_pct}% of cells)."
        )
    if duplicates > 0:
        summary_parts.append(f"{duplicates:,} duplicate rows were identified and flagged.")

    # --- Key insights (narrative)
    if insights:
        summary_parts.append("Key findings from automated analysis:")
        for ins in insights[:3]:
            title = ins.get("title", "")
            detail = ins.get("details", "")
            if title and detail:
                summary_parts.append(f"\u2022 {title}: {detail}")

    # --- ML Prediction outcome
    pred_msg = prediction_data.get("message", "")
    pred_acc = prediction_data.get("accuracy", 0)
    if pred_acc and pred_acc > 0:
        target = prediction_data.get("target", "target variable")
        model = prediction_data.get("model", "machine learning model")
        summary_parts.append(
            f"Predictive modelling ({model}) achieved {pred_acc}% accuracy on '{target}'."
        )
    elif pred_msg:
        summary_parts.append(f"Predictive analysis note: {pred_msg}")

    # --- Top recommendation
    if recommendations:
        top_rec = recommendations[0]
        action = top_rec.get("action", "")
        impact = top_rec.get("impact", "")
        if action and impact:
            summary_parts.append(
                f"Top recommended action ({impact} impact): {action}."
            )

    summary = " ".join(summary_parts)

    return {
        "summary": summary,
        "cleaningLog": cleaning_data.get("cleaningLog", []),
        "insights": insights,
        "recommendations": recommendations
    }