"""
chat_agent.py — InsightFlow Dataset Chat Agent

Architecture:
  - LLMProvider abstract base → swap providers without touching routing logic
  - RuleBasedProvider (default, no external API required)
  - Future: OpenAIProvider / GeminiProvider inherit from LLMProvider

Every response is a structured dict:
  { "answer": str, "source": str, "intent": str, "confidence": float }

The /chat FastAPI endpoint returns this dict directly — the `answer` key
preserves backward compatibility with existing frontend consumers.
"""

from __future__ import annotations
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from typing import Dict, Any
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("GROQ KEY =", os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────────────────────────
# LLM Provider abstraction  (plug in any LLM without changing routing logic)
# ─────────────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base for all LLM / rule-based backends."""

    @abstractmethod
    def complete(self, question: str, context: Dict[str, Any]) -> str:
        """Return a markdown-safe answer string."""
        ...


class RuleBasedProvider(LLMProvider):
    """
    Default provider — pure rule-based routing, no external API needed.
    When you wire in an LLM, create a subclass (e.g. GeminiProvider) and
    pass it to get_chat_response() via the `provider` parameter.
    """

    def complete(self, question: str, context: Dict[str, Any]) -> str:
        return _rule_based_router(question, context)

class GroqProvider(LLMProvider):
    def complete(self, question: str, context: Dict[str, Any]) -> str:
        prompt = f"""
You are InsightFlow AI, an expert Senior Data Analyst and Strategy Consultant.

Dataset Context:
- Name: {context.get("name", "Unknown")}
- Shape: {context.get("rows", "?")} rows, {context.get("cols", "?")} columns

Key Data Statistics:
{context.get("stats")}

Discovered Insights:
{context.get("insights")}

Machine Learning Predictions:
{context.get("predictions")}

Automated Report Summary:
{context.get("report")}

User Question:
{question}

Instructions:
1. Provide a highly analytical, insight-driven response.
2. Synthesize the provided stats, insights, and predictions into actionable business intelligence where relevant.
3. Be concise, format with clean markdown, and do not use generic filler words. Provide direct value.
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a senior data science consultant focusing on actionable insights and rigorous analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Groq chat failed, falling back to rule-based:", e)
            return _rule_based_router(question, context)

# ─────────────────────────────────────────────────────────────────────────────
# Intent classifier  (returns intent label + confidence, 0–1)
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_KEYWORDS: list[tuple[str, list[str], float]] = [
    # (intent_label, trigger_keywords, confidence)
    ("greeting",      ["hello", "hi ", "hey", "help", "what can you do", "capabilities", "capabilities"], 0.95),
    ("row_count",     ["how many rows", "row count", "record count", "number of rows", "total rows", "size of"], 0.98),
    ("col_count",     ["how many column", "column count", "number of column", "how many features", "feature count"], 0.98),
    ("list_columns",  ["column names", "list column", "what are the column", "headers", "fields", "attributes", "what columns"], 0.95),
    ("missing",       ["missing", "null", "na values", "nan", "empty cells", "incomplete", "nulls"], 0.97),
    ("duplicates",    ["duplicate", "repeated rows", "duplicates"], 0.97),
    ("quality",       ["quality", "data health", "how clean", "cleanliness"], 0.92),
    ("dataset_type",  ["dataset type", "what type", "type of data", "what kind", "category of data"], 0.90),
    ("insights",      ["insight", "finding", "key finding", "tell me something", "interesting", "analysis result", "what did you find"], 0.93),
    ("recommendations", ["recommend", "suggestion", "what should i", "next step", "improve", "advise", "business action"], 0.93),
    ("predictions",   ["predict", "model", "accuracy", "ml", "machine learning", "classifier", "target variable", "forecast", "churn", "risk"], 0.95),
    ("summary",       ["summary", "overview", "tell me about", "describe", "what is this", "give me a", "summarize", "explain this dataset"], 0.92),
    ("correlations",  ["correlat", "relationship", "related", "depend", "association", "co-vary", "linked"], 0.93),
    ("data_types",    ["data type", "dtype", "column type", "what type is", "numeric column", "categorical column", "type of column"], 0.94),
    ("top_values",    ["top values", "most common", "frequent", "most frequent", "value count", "distribution of", "unique value"], 0.90),
    ("charts",        ["chart", "graph", "visualization", "plot", "what charts", "what graphs", "generated chart"], 0.88),
    ("cleaning",      ["clean", "cleaning log", "what was cleaned", "preprocessing", "imputed"], 0.90),
    ("explain_col",   ["explain", "tell me about column", "what is", "what does column", "describe column"], 0.80),
]


def _detect_intent(question: str) -> tuple[str, float]:
    """Return (intent_label, confidence) for a given question string."""
    q = question.lower().strip()
    for label, kws, conf in _INTENT_KEYWORDS:
        if any(kw in q for kw in kws):
            return label, conf
    return "general", 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based response router
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_router(question: str, ctx: Dict[str, Any]) -> str:
    if not ctx:
        return (
            "No dataset is currently loaded. "
            "Please upload a CSV or Excel file first, then ask me anything about it."
        )

    q = question.lower().strip()

    # ── Extract context fields ─────────────────────────────────────────────
    name             = ctx.get("name", "the dataset")
    rows             = ctx.get("rows", 0)
    cols             = ctx.get("cols", 0)
    headers: list    = ctx.get("headers", [])
    stats: dict      = ctx.get("stats", {})
    insights: list   = ctx.get("insights", [])
    recommendations: list = ctx.get("recommendations", [])
    predictions: dict = ctx.get("predictions", {})
    report: dict     = ctx.get("report", {})
    column_info: dict = ctx.get("columnInfo", {})
    dataset_type: str = ctx.get("datasetType", "Generic")
    column_profiles: list = ctx.get("columnProfiles", [])
    charts_data: list = ctx.get("chartsData", [])

    missing   = stats.get("missingValues", 0)
    dupes     = stats.get("duplicateRows", 0)
    quality   = stats.get("qualityScore", 0)
    numeric_cols: list   = column_info.get("numericColumns", [])
    categorical_cols: list = column_info.get("categoricalColumns", [])

    def _has(*kws: str) -> bool:
        return any(kw in q for kw in kws)

    # ── Greeting / capabilities ────────────────────────────────────────────
    if _has("hello", "hi ", "hey", "help", "what can you do", "capabilities"):
        return (
            f"👋 Hello! I'm **InsightFlow AI Copilot**, connected to **{name}** "
            f"({rows:,} rows · {cols} columns · {dataset_type}).\n\n"
            "Here's what you can ask me:\n\n"
            "| Category | Example Question |\n"
            "|---|---|\n"
            "| 📊 Overview | *Summarize this dataset* |\n"
            "| 🔢 Structure | *How many rows and columns?* |\n"
            "| ❓ Quality | *Are there missing values?* |\n"
            "| 🔗 Correlations | *What columns are correlated?* |\n"
            "| ⭐ Insights | *What are the key findings?* |\n"
            "| 💼 Actions | *What should I do next?* |\n"
            "| 🤖 ML Model | *Was a prediction model trained?* |\n"
            "| 📐 Column Info | *Tell me about the Age column* |\n\n"
            "What would you like to know?"
        )

    # ── Row count ──────────────────────────────────────────────────────────
    if _has("how many rows", "row count", "record count", "number of rows",
            "how many records", "total rows", "total records", "size of the dataset"):
        return f"**{name}** contains **{rows:,} rows** (records)."

    # ── Column count ───────────────────────────────────────────────────────
    if _has("how many column", "column count", "number of column",
            "how many features", "feature count", "how many fields"):
        return (
            f"**{name}** has **{cols} columns** total:\n\n"
            f"- 🔢 **Numeric:** {len(numeric_cols)} columns\n"
            f"- 🔤 **Categorical:** {len(categorical_cols)} columns"
        )

    # ── List columns ───────────────────────────────────────────────────────
    if _has("column names", "list column", "what are the column",
            "headers", "fields", "attributes", "what columns"):
        num_preview = ", ".join(f"`{c}`" for c in numeric_cols[:10]) or "None"
        cat_preview = ", ".join(f"`{c}`" for c in categorical_cols[:10]) or "None"
        extra_num = f" (+{len(numeric_cols)-10} more)" if len(numeric_cols) > 10 else ""
        extra_cat = f" (+{len(categorical_cols)-10} more)" if len(categorical_cols) > 10 else ""
        return (
            f"**{name}** — {cols} columns:\n\n"
            f"**Numeric ({len(numeric_cols)}):** {num_preview}{extra_num}\n\n"
            f"**Categorical ({len(categorical_cols)}):** {cat_preview}{extra_cat}"
        )

    # ── Data types ─────────────────────────────────────────────────────────
    if _has("data type", "dtype", "column type", "what type is",
            "numeric column", "categorical column", "type of column"):
        lines = ["**Column Types Overview:**\n"]
        lines.append("| Column | Type |")
        lines.append("|---|---|")
        for h in headers[:20]:
            col_type = "🔢 Numeric" if h in numeric_cols else "🔤 Categorical"
            lines.append(f"| `{h}` | {col_type} |")
        if len(headers) > 20:
            lines.append(f"| ... | *+{len(headers)-20} more* |")
        return "\n".join(lines)

    # ── Missing values ─────────────────────────────────────────────────────
    if _has("missing", "null", "na values", "nan", "empty cells", "incomplete", "nulls"):
        if missing == 0:
            return (
                f"✅ **No missing values** were found in **{name}**.\n\n"
                "The dataset is 100% complete — no imputation was needed."
            )
        pct = round((missing / (rows * cols)) * 100, 2) if rows * cols > 0 else 0

        # Build per-column missing breakdown from profiles
        missing_cols = [
            p for p in column_profiles if p.get("missing", 0) > 0
        ]
        breakdown = ""
        if missing_cols:
            breakdown = "\n\n**Missing by Column:**\n"
            for p in missing_cols[:8]:
                col_missing = p.get("missing", 0)
                col_pct = round(col_missing / rows * 100, 1) if rows > 0 else 0
                breakdown += f"- `{p['column']}`: {col_missing:,} missing ({col_pct}%)\n"
            if len(missing_cols) > 8:
                breakdown += f"- *...and {len(missing_cols) - 8} more columns*\n"

        return (
            f"**{name}** had **{missing:,} missing values** ({pct}% of all cells). "
            f"These were automatically imputed during cleaning using:\n\n"
            f"- **Numeric columns** → median imputation\n"
            f"- **Categorical columns** → mode imputation"
            + breakdown
        )

    # ── Duplicates ─────────────────────────────────────────────────────────
    if _has("duplicate", "duplicates", "repeated rows"):
        if dupes == 0:
            return f"✅ **No duplicate rows** were found in **{name}**."
        return (
            f"**{dupes:,} duplicate rows** were detected and removed from **{name}** "
            "during the data cleaning phase.\n\n"
            f"After deduplication: **{rows:,} unique records** remain."
        )

    # ── Quality score ──────────────────────────────────────────────────────
    if _has("quality", "data health", "how clean", "cleanliness", "score"):
        if quality >= 85:
            rating, icon = "Excellent", "🟢"
        elif quality >= 65:
            rating, icon = "Good", "🟡"
        else:
            rating, icon = "Needs Attention", "🔴"
        return (
            f"**Data Quality Score: {quality}/100** {icon} {rating}\n\n"
            f"| Metric | Value |\n"
            f"|---|---|\n"
            f"| Missing Values | {missing:,} |\n"
            f"| Duplicate Rows Removed | {dupes:,} |\n"
            f"| Rows | {rows:,} |\n"
            f"| Columns | {cols} |\n\n"
            f"**Rating:** {rating} — "
            + ("The dataset is in great shape for analysis." if quality >= 85
               else "Consider reviewing columns with high missing rates." if quality >= 65
               else "Significant data quality issues detected. Review missing and duplicate data carefully.")
        )

    # ── Dataset type ───────────────────────────────────────────────────────
    if _has("dataset type", "what type", "type of data", "what kind", "category of"):
        return (
            f"**{name}** is classified as a **{dataset_type}** dataset "
            f"based on its column structure and content patterns."
        )

    # ── Correlations ───────────────────────────────────────────────────────
    if _has("correlat", "relationship", "related", "depend", "association", "co-vary", "linked"):
        if len(numeric_cols) < 2:
            return (
                f"**{name}** has fewer than 2 numeric columns, "
                "so correlation analysis is not applicable."
            )
        # Attempt to find correlation hints from chartsData heatmap
        heatmap_data = next(
            (c.get("data", []) for c in charts_data if c.get("type") == "heatmap"), []
        )
        if heatmap_data:
            strong = [
                d for d in heatmap_data
                if isinstance(d.get("val"), (int, float)) and abs(d["val"]) >= 0.5 and d["x"] != d["y"]
            ]
            strong_sorted = sorted(strong, key=lambda x: abs(x["val"]), reverse=True)[:6]
            if strong_sorted:
                lines = ["**Notable Correlations (|r| ≥ 0.5):**\n"]
                lines.append("| Column A | Column B | Correlation |\n|---|---|---|")
                for d in strong_sorted:
                    val = d["val"]
                    direction = "↑ Positive" if val > 0 else "↓ Negative"
                    strength = "Strong" if abs(val) >= 0.7 else "Moderate"
                    lines.append(f"| `{d['x']}` | `{d['y']}` | {val:+.2f} ({strength} {direction}) |")
                lines.append(
                    f"\n💡 *Correlation is computed on {len(numeric_cols)} numeric columns. "
                    "Values close to ±1.0 indicate strong linear relationships.*"
                )
                return "\n".join(lines)

        return (
            f"**{name}** has **{len(numeric_cols)} numeric columns**: "
            f"{', '.join(f'`{c}`' for c in numeric_cols[:8])}.\n\n"
            "A correlation heatmap has been generated in the **Full Analysis** tab. "
            "Look for values close to **+1.0** (positive) or **−1.0** (negative) for strong relationships.\n\n"
            "💡 *Tip: Ask about a specific pair — e.g. \"How are Age and Fare related?\"*"
        )

    # ── Top values / distribution ──────────────────────────────────────────
    if _has("top values", "most common", "frequent", "most frequent",
            "value count", "unique value"):
        # Check if a specific column name is mentioned
        for col in headers:
            if col.lower() in q:
                profile = next((p for p in column_profiles if p.get("column") == col), None)
                if profile:
                    if profile.get("metricType") == "Categorical":
                        top = profile.get("topValues", [])
                        if top:
                            lines = [f"**Top values in `{col}`:**\n", "| Value | Count |", "|---|---|"]
                            for tv in top[:8]:
                                lines.append(f"| {tv.get('value', '?')} | {tv.get('count', '?'):,} |")
                            return "\n".join(lines)
                    else:
                        return (
                            f"`{col}` is a **numeric column** — value counts are not directly applicable.\n\n"
                            f"**Stats:** min={profile.get('min','?')}, "
                            f"median={profile.get('median','?')}, "
                            f"max={profile.get('max','?')}, "
                            f"std={profile.get('stdDev','?')}"
                        )

        # No column specified — show top categorical columns
        cat_profiles = [p for p in column_profiles if p.get("metricType") == "Categorical"][:3]
        if not cat_profiles:
            return "No categorical columns found to show top values for."
        lines = ["**Most frequent values per categorical column:**\n"]
        for p in cat_profiles:
            top = p.get("topValues", [])
            if top:
                top_str = ", ".join(
                    f"`{tv['value']}` ({tv['count']:,})" for tv in top[:3]
                )
                lines.append(f"- **`{p['column']}`**: {top_str}")
        return "\n".join(lines)

    # ── Charts / visualizations ────────────────────────────────────────────
    if _has("chart", "graph", "visualization", "plot", "what charts", "generated"):
        if not charts_data:
            return (
                "No charts have been generated yet. "
                "Upload a dataset and run the full analysis to auto-generate visualizations."
            )
        chart_types = {}
        for c in charts_data:
            ct = c.get("type", "unknown")
            chart_types[ct] = chart_types.get(ct, 0) + 1

        lines = [f"**{len(charts_data)} charts were generated** for **{name}**:\n"]
        for ct, cnt in chart_types.items():
            emoji = {
                "histogram": "📊", "bar": "📊", "pie": "🥧",
                "scatter": "⚡", "heatmap": "🌡️", "line": "📈",
                "box": "📦", "area": "🌊", "grouped-bar": "📊",
            }.get(ct, "📉")
            plural = "chart" if cnt == 1 else "charts"
            lines.append(f"- {emoji} **{ct.title()}**: {cnt} {plural}")

        lines.append(
            "\n💡 *Visit the **Full Analysis** tab for interactive charts, "
            "or use the **Analysis Workspace** to build custom visualizations.*"
        )
        return "\n".join(lines)

    # ── Insights ───────────────────────────────────────────────────────────
    if _has("insight", "finding", "key finding", "tell me something",
            "interesting", "analysis result", "what did you find"):
        if not insights:
            return "No insights available yet. Upload and analyze a dataset to get AI-generated insights."
        lines = ["**🔍 Top AI-Generated Insights:**\n"]
        for i, ins in enumerate(insights[:5], 1):
            title   = ins.get("title", "")
            details = ins.get("details", "")
            metric  = ins.get("metric", "")
            lines.append(f"**{i}. {title}**")
            if details:
                lines.append(details)
            if metric:
                lines.append(f"*📊 {metric}*")
            lines.append("")
        return "\n".join(lines)

    # ── Recommendations ────────────────────────────────────────────────────
    if _has("recommend", "suggestion", "action", "what should i",
            "next step", "improve", "advise", "business action"):
        if not recommendations:
            return "No recommendations available yet."
        lines = ["**💼 Actionable Recommendations:**\n"]
        for i, r in enumerate(recommendations[:4], 1):
            action  = r.get("action", "")
            impact  = r.get("impact", "")
            outcome = r.get("expectedOutcome", "")
            lines.append(f"**{i}. {action}**")
            if impact:
                lines.append(f"*Impact: {impact}*")
            if outcome:
                lines.append(outcome)
            lines.append("")
        return "\n".join(lines)

    # ── ML predictions ─────────────────────────────────────────────────────
    if _has("predict", "model", "accuracy", "ml", "machine learning",
            "classifier", "target variable", "forecast", "churn", "risk"):
        acc    = predictions.get("accuracy", 0)
        target = predictions.get("targetVariable")
        avail  = predictions.get("available", bool(acc and acc > 0))
        if not avail or not target:
            return (
                "🤖 **No predictive model was trained** for this dataset.\n\n"
                "A suitable binary/categorical target column was not detected. "
                "Try uploading a labeled dataset with a column like `survived`, `churn`, `is_canceled`, etc."
            )
        best   = predictions.get("bestModel", {})
        models = predictions.get("modelsEvaluated", [])
        lines  = [
            f"🤖 **Prediction Results — {name}:**\n",
            f"- **Target variable:** `{target}`",
            f"- **Problem type:** {predictions.get('problemType', 'Classification')}",
            f"- **Best model:** {best.get('name', 'N/A')}",
            f"- **Accuracy:** {acc}%",
            f"- **High-risk cases:** {predictions.get('highRiskCases', 0):,}",
        ]
        if models:
            lines.append("\n**Models Evaluated:**\n")
            lines.append("| Model | Accuracy | Status |")
            lines.append("|---|---|---|")
            for m in models:
                flag = "✅ Best" if m.get("isBest") else ""
                lines.append(f"| {m['name']} | {m['accuracy']}% | {flag} |")
        return "\n".join(lines)

    # ── Summary / overview ─────────────────────────────────────────────────
    if _has("summary", "overview", "tell me about", "describe",
            "what is this", "give me a", "summarize", "explain this dataset"):
        summary_text = report.get("summary", "")
        quality_label = "Excellent 🟢" if quality >= 85 else "Good 🟡" if quality >= 65 else "Needs Attention 🔴"
        base = (
            f"**📋 {name} — Dataset Overview**\n\n"
            f"| Property | Value |\n"
            f"|---|---|\n"
            f"| Rows | {rows:,} |\n"
            f"| Columns | {cols} |\n"
            f"| Dataset Type | {dataset_type} |\n"
            f"| Quality Score | {quality}/100 ({quality_label}) |\n"
            f"| Missing Values | {missing:,} |\n"
            f"| Duplicates Removed | {dupes:,} |\n"
            f"| Numeric Columns | {len(numeric_cols)} |\n"
            f"| Categorical Columns | {len(categorical_cols)} |\n"
        )
        if summary_text:
            base += f"\n**Summary:**\n{summary_text}"
        return base

    # ── Column-specific detail (explain / tell me about <col>) ─────────────
    for profile in column_profiles:
        col_name = profile.get("column", "")
        if col_name.lower() in q:
            metric_type = profile.get("metricType", "Unknown")
            lines = [f"**Column: `{col_name}` ({metric_type})**\n"]
            if metric_type == "Numeric":
                lines += [
                    f"| Stat | Value |",
                    f"|---|---|",
                    f"| Mean | {profile.get('mean', 'N/A')} |",
                    f"| Median | {profile.get('median', 'N/A')} |",
                    f"| Min | {profile.get('min', 'N/A')} |",
                    f"| Max | {profile.get('max', 'N/A')} |",
                    f"| Std Dev | {profile.get('stdDev', 'N/A')} |",
                    f"| Missing | {profile.get('missing', 0):,} |",
                ]
            else:
                top = profile.get("topValues", [])
                top_str = ", ".join(f"`{tv['value']}`" for tv in top[:5]) if top else "N/A"
                lines += [
                    f"| Stat | Value |",
                    f"|---|---|",
                    f"| Unique Values | {profile.get('median', 'N/A')} |",
                    f"| Most Frequent | {profile.get('mean', 'N/A')} |",
                    f"| Missing | {profile.get('missing', 0):,} |",
                    f"| Top Values | {top_str} |",
                ]
            return "\n".join(lines)

    # ── Cleaning log ───────────────────────────────────────────────────────
    if _has("clean", "cleaning log", "what was cleaned", "preprocessing", "imputed"):
        cleaning_log = report.get("cleaningLog", [])
        if not cleaning_log:
            return "No cleaning log available for this dataset."
        lines = ["**🧹 Data Cleaning Log:**\n"]
        for step in cleaning_log:
            lines.append(f"- {step}")
        return "\n".join(lines)

    # ── Generic fallback ───────────────────────────────────────────────────
    return (
        f"I'm connected to **{name}** ({rows:,} rows · {cols} columns · {dataset_type}).\n\n"
        "I didn't quite catch what you're looking for. Try one of these:\n\n"
        "- *Summarize this dataset*\n"
        "- *What columns are correlated?*\n"
        "- *Are there missing values?*\n"
        "- *What are the key insights?*\n"
        "- *Was a prediction model trained?*\n"
        "- *Tell me about the `<column_name>` column*\n\n"
        "What would you like to know?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PROVIDER = RuleBasedProvider()
_LLM_PROVIDER = GroqProvider()


def get_chat_response(
    question: str,
    ctx: Dict[str, Any],
    provider: LLMProvider | None = None,
) -> dict:
    selected_provider = provider or _DEFAULT_PROVIDER

    intent, confidence = _detect_intent(question)
    llm_intents = ["insights", "recommendations", "summary", "general"]

    if intent in llm_intents:
        selected_provider = _LLM_PROVIDER

    try:
        answer = selected_provider.complete(question, ctx)

        source = (
            "rule_based"
            if isinstance(selected_provider, RuleBasedProvider)
            else type(selected_provider).__name__.lower().replace("provider", "")
        )

    except Exception as exc:
        answer = (
            f"Error: {exc}\n\n"
            "Please try again."
        )
        source = "error"
        confidence = 0.0

    return {
        "answer": answer,
        "source": source,
        "intent": intent,
        "confidence": round(confidence, 2),
    }
