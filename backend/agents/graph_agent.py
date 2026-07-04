import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_best_graph(dataset_context, query):
    prompt = f"""
You are an expert data visualization agent.

Choose TOP 4 BEST graphs for analysis.

Prioritize:
1. Insight quality
2. Statistical correctness
3. Human readability

Dataset Context:
{dataset_context}

User Query:
{query}

Available graph types:
bar, line, pie, scatter, histogram, boxplot, heatmap

Important:
- Never choose scatter for binary vs numerical
- Avoid pie charts unless categories <= 5
- Prefer boxplot for numerical vs categorical
- Prefer histogram for distributions
- Prefer heatmap for correlations
- Prefer meaningful graphs over visually attractive graphs

Return ONLY a valid JSON array:

[
  {{
    "chartType": "",
    "xAxis": "",
    "yAxis": "",
    "title": ""
  }}
]
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a data visualization expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )

        content = response.choices[0].message.content
        print("RAW RESPONSE:", repr(content))

        if not content or not content.strip():
            return []

        content = content.strip()

        if content.startswith("```json"):
            content = content.replace("```json", "", 1)

        if content.startswith("```"):
            content = content.replace("```", "", 1)

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        parsed = json.loads(content)

        if isinstance(parsed, dict):
            parsed = [parsed]

        return parsed

    except Exception as e:
        print("Graph agent failed:", e)
        return []