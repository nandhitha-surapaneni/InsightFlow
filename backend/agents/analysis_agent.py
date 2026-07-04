import pandas as pd
import numpy as np
import os
import json
import logging
from groq import Groq

# Initialize Groq client
_api_key = os.getenv("GROQ_API_KEY")
_client = Groq(api_key=_api_key) if _api_key else None

# ─────────────────────────────────────────────
# Business Domain Detection
# ─────────────────────────────────────────────
def detect_dataset_type(df: pd.DataFrame) -> str:
    cols = " ".join([c.lower() for c in df.columns])
    
    if any(k in cols for k in ["hotel", "booking", "reservation", "resort", "stays"]):
        return "Hotel & Hospitality"
    elif any(k in cols for k in ["sales", "revenue", "order", "discount", "product", "price"]):
        return "Sales & E-Commerce"
    elif any(k in cols for k in ["employee", "salary", "attrition", "hire", "department"]):
        return "HR & Talent"
    elif any(k in cols for k in ["loan", "default", "balance", "credit", "transaction"]):
        return "Finance & Banking"
    elif any(k in cols for k in ["patient", "diagnosis", "hospital", "treatment", "disease"]):
        return "Healthcare"
    return "Generic Business"

# ─────────────────────────────────────────────
# Helper: Find Target Business KPI
# ─────────────────────────────────────────────
def find_target_kpi(df: pd.DataFrame, numeric_cols: list) -> str:
    for col in numeric_cols:
        c = col.lower()
        if any(k in c for k in ['revenue', 'sales', 'profit', 'price', 'fare', 'amount', 'salary', 'total', 'margin', 'marks', 'score']):
            return col
    return numeric_cols[0] if numeric_cols else None

# ─────────────────────────────────────────────
# Core Logic: Generate LLM Insights
# ─────────────────────────────────────────────
def generate_insights_and_recommendations_llm(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """
    Generates dataset-specific insights and recommendations using Groq LLM.
    Returns a dict with 'insights' and 'recommendations' keys.
    """
    # 1. Extract statistical facts
    facts = []
    facts.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")
    
    target_kpi = find_target_kpi(df, numeric_cols)
    if target_kpi:
        facts.append(f"Primary target/KPI appears to be '{target_kpi}' (mean: {df[target_kpi].mean():.2f}).")
        
    # Correlations
    if len(numeric_cols) >= 2:
        try:
            sample_df = df[numeric_cols].dropna()
            if len(sample_df) > 5000:
                sample_df = sample_df.sample(5000, random_state=42)
            corr = sample_df.corr()
            
            ignore = ['id', 'index', 'serial', 'key', 'uuid', 'code', 'rownum']
            valid_idx = [c for c in corr.columns if not any(i in str(c).lower() for i in ignore)]
            corr = corr.loc[valid_idx, valid_idx]
            
            # Unstack and filter
            corr_unstacked = corr.unstack().reset_index()
            corr_unstacked.columns = ['var1', 'var2', 'corr']
            corr_unstacked = corr_unstacked[corr_unstacked['var1'] < corr_unstacked['var2']]
            corr_unstacked['abs_corr'] = corr_unstacked['corr'].abs()
            
            top_corrs = corr_unstacked.sort_values(by='abs_corr', ascending=False).head(3)
            for _, row in top_corrs.iterrows():
                if row['abs_corr'] > 0.2:
                    direction = "positive" if row['corr'] > 0 else "negative"
                    facts.append(f"Strong {direction} correlation between '{row['var1']}' and '{row['var2']}' ({row['corr']:.2f}).")
        except Exception:
            pass

    # Grouped means for target_kpi
    valid_cats = [c for c in categorical_cols if 2 <= df[c].nunique() <= 15]
    if valid_cats and target_kpi:
        for cat in valid_cats[:2]: # limit to top 2 categorical columns
            try:
                grouped = df.groupby(cat)[target_kpi].mean().sort_values(ascending=False)
                counts = df[cat].value_counts()
                valid_groups = grouped[counts >= len(df) * 0.05] # at least 5% of data
                if not valid_groups.empty and len(valid_groups) >= 2:
                    top_cat = valid_groups.index[0]
                    top_val = valid_groups.iloc[0]
                    bottom_cat = valid_groups.index[-1]
                    bottom_val = valid_groups.iloc[-1]
                    facts.append(f"For categorical column '{cat}', '{top_cat}' has the highest average {target_kpi} ({top_val:.2f}), while '{bottom_cat}' has the lowest ({bottom_val:.2f}).")
            except Exception:
                pass

    # Binary columns distributions
    binary_cols = [c for c in df.columns if df[c].nunique() == 2]
    if binary_cols:
        for bcol in binary_cols[:2]:
            try:
                vc = df[bcol].value_counts(normalize=True)
                top_cls = vc.index[0]
                top_pct = vc.iloc[0] * 100
                facts.append(f"Column '{bcol}' is highly skewed towards '{top_cls}' ({top_pct:.1f}%).")
            except Exception:
                pass
                
    facts_text = "\n".join(f"- {f}" for f in facts)
    
    # Define fallback function
    def _fallback():
        insights = []
        recs = []
        
        # Build simple insights from facts
        for idx, f in enumerate(facts[:3]):
            metric = "Behavioral" if "correlation" in f else "Opportunity" if "average" in f else "Overview"
            impact = "High" if metric == "Opportunity" else "Medium"
            
            insights.append({
                "title": f"Key Finding {idx+1}",
                "details": f,
                "metric": metric,
                "impact": impact
            })
            
            if "correlation" in f:
                recs.append({
                    "action": "Investigate relationship further",
                    "impact": "Medium",
                    "expectedOutcome": "Understanding the link between these correlated variables can yield predictive value."
                })
            elif "average" in f:
                recs.append({
                    "action": "Focus on top-performing groups",
                    "impact": "High",
                    "expectedOutcome": f"Targeting segments with the highest {target_kpi or 'metric'} will maximize return on investment."
                })
            else:
                recs.append({
                    "action": "Review data distribution",
                    "impact": "Low",
                    "expectedOutcome": "Addressing imbalances or skewness can improve model reliability and operational understanding."
                })
                
        # Fill if less than 3
        while len(insights) < 3:
            insights.append({"title": "General Insight", "details": "The dataset contains standard operational records.", "metric": "Overview", "impact": "Low"})
        while len(recs) < 3:
            recs.append({"action": "Continue regular monitoring", "impact": "Low", "expectedOutcome": "Maintains current baseline performance."})
            
        return {"insights": insights[:3], "recommendations": recs[:3]}

    if not _client:
        return _fallback()
        
    prompt = f"""
You are an expert data analyst. You are provided with statistical facts extracted from a dataset.
Your task is to generate EXACTLY 3 insights and 3 actionable recommendations based ONLY on these specific facts.

Do NOT use generic corporate jargon (e.g. "Optimize operational efficiency", "Conduct A/B testing") unless it directly relates to the facts.
Always ask yourself: "What important story does this dataset tell based on the numbers?"

Statistical Facts:
{facts_text}

Output format must be a raw JSON object with the following structure:
{{
  "insights": [
    {{
      "title": "Short title of the finding",
      "details": "The finding and why it matters (use the specific numbers from the facts).",
      "metric": "One word tag (e.g. Risk, Opportunity, Behavioral, Trend, Performance)",
      "impact": "High" | "Medium" | "Low"
    }}
  ],
  "recommendations": [
    {{
      "action": "Specific action to take based on the insights",
      "impact": "High" | "Medium" | "Low",
      "expectedOutcome": "The expected impact of this action."
    }}
  ]
}}

Ensure there are exactly 3 items in the "insights" array and 3 items in the "recommendations" array.
Return ONLY raw JSON. No markdown wrappers like ```json.
"""

    try:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a senior data scientist. Respond with raw JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].startswith("```"): lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        parsed = json.loads(content)
        
        insights = parsed.get("insights", [])
        recommendations = parsed.get("recommendations", [])
        
        if len(insights) < 3 or len(recommendations) < 3:
            # fill with fallbacks if LLM gave partial response
            fallback_res = _fallback()
            while len(insights) < 3:
                insights.append(fallback_res["insights"][len(insights)])
            while len(recommendations) < 3:
                recommendations.append(fallback_res["recommendations"][len(recommendations)])
                
        return {
            "insights": insights[:3],
            "recommendations": recommendations[:3]
        }
    except Exception as e:
        logging.getLogger(__name__).error(f"LLM insights generation failed: {e}")
        return _fallback()


# ─────────────────────────────────────────────
# Dynamic Chart Data Generation
# ─────────────────────────────────────────────
def generate_chartsData(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> list:
    from agents.graph_llm_agent import recommend_dashboard_graphs_llm
    from agents.chart_agent import generate_custom_chart

    recommended_configs = recommend_dashboard_graphs_llm(df)
    charts = []

    for config in recommended_configs:
        try:
            chart = generate_custom_chart(
                df=df,
                x_axis=config["xAxis"],
                y_axis=config.get("yAxis"),
                chart_type=config["chartType"],
                aggregation=config.get("aggregation", "count") or "count"
            )
            if config.get("title"):
                chart["title"] = config["title"]
            charts.append(chart)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to build chart {config}: {e}")

    if not charts:
        charts.append({
            "chartType": "bar",
            "title": "Business Overview",
            "data": [
                {"category": "Total Records", "value": len(df)},
                {"category": "Tracked Variables", "value": len(df.columns)}
            ],
            "xAxis": df.columns[0],
            "yAxis": None,
            "xLabel": "Metric",
            "yLabel": "Count"
        })

    for chart in charts:
        if "chartType" in chart:
            chart["type"] = chart["chartType"]

    return charts

# ─────────────────────────────────────────────
# Column Profiles
# ─────────────────────────────────────────────
def generate_column_profiles(df):
    profiles = []

    for col in df.columns:
        profile = {
            "column": col,
            "missing": int(df[col].isnull().sum())
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            clean_series = df[col].dropna()
            profile.update({
                "metricType": "Numeric",
                "mean": round(float(clean_series.mean()), 2) if not clean_series.empty else 0,
                "median": round(float(clean_series.median()), 2) if not clean_series.empty else 0,
                "min": round(float(clean_series.min()), 2) if not clean_series.empty else 0,
                "max": round(float(clean_series.max()), 2) if not clean_series.empty else 0,
                "stdDev": round(float(clean_series.std()), 2) if not clean_series.empty else 0
            })
        else:
            unique_count = int(df[col].nunique())
            mode_val = str(df[col].mode().iloc[0]) if not df[col].mode().empty else "N/A"
            profile.update({
               "metricType": "Categorical",
               "mostFrequent": mode_val,
               "uniqueValues": unique_count,
               "mean": None,
               "median": None,
               "min": None,
               "max": None,
               "stdDev": None
            })

        profiles.append(profile)

    return profiles

# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────
def analyze_data(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    total_columns = len(df.columns)
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Auto-detect datetime columns for trend analysis
    dt_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if not dt_cols:
        # Fallback check for string dates
        for col in categorical_cols:
            if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
                dt_cols.append(col)

    dataset_type = detect_dataset_type(df)

    # Use the new LLM function for dataset-specific insights
    llm_results = generate_insights_and_recommendations_llm(df, numeric_cols, categorical_cols)
    insights = llm_results.get("insights", [])
    recommendations = llm_results.get("recommendations", [])
    
    chartsData = generate_chartsData(df, numeric_cols, categorical_cols)
    columnProfiles = generate_column_profiles(df)

    return {
        "stats": {
            "totalRows": total_rows,
            "totalColumns": total_columns,
            "missingValues": missing_values,
            "duplicateRows": duplicate_rows
        },
        "datasetType": dataset_type,
        "columnInfo": {
            "numericColumns": numeric_cols,
            "categoricalColumns": categorical_cols,
        },
        "insights": insights,
        "recommendations": recommendations,
        "chartsData": chartsData,
        "columnProfiles": columnProfiles,
    }
