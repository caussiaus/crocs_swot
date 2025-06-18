#!/usr/bin/env python3
"""
C_make_dashboard_payload.py
─────────────────────────
Generate events.json + bulletpoints.json for the SWOT quadrant
dashboard, using the yearly SWOT summaries from B_yearly_swot_with_mistral.py.

This is the final step in the pipeline that creates the visualization data.
"""

from __future__ import annotations
import json, itertools, textwrap
from pathlib import Path
import pandas as pd
from transformers import pipeline

# ── paths ──────────────────────────────────────────────────────────────────
SWOT_JSON = Path("yearly_swot/yearly_swot.json")  # Output from B step
DASHBOARD_DIR = Path("dashboard")
DASHBOARD_DIR.mkdir(exist_ok=True)

# ── 1. load yearly SWOT data ───────────────────────────────────────────────
if not SWOT_JSON.exists():
    raise FileNotFoundError(f"SWOT data not found at {SWOT_JSON}. Run B_yearly_swot_with_mistral.py first.")

with open(SWOT_JSON, 'r') as f:
    swot_data = json.load(f)

# ── 2. convert to dataframe for processing ─────────────────────────────────
rows = []
for year_str, categories in swot_data.items():
    year = int(year_str)
    for category, data in categories.items():
        rows.append({
            'year': year,
            'label': category,
            'count': data['count'],
            'summary': data['summary']
        })

df = pd.DataFrame(rows)
max_cnt = df["count"].max()

# ── 3. map into dashboard schema ────────────────────────────────────────────
quad_xy = {"Strength":(.25,.25), "Weakness":(.75,.25), "Opportunity":(.25,.75), "Threat":(.75,.75)}
color   = {"Strength":"46,125,50", "Opportunity":"46,125,50", "Weakness":"229,57,53", "Threat":"229,57,53"}

events  = []
bullets = []

for _, row in df.iterrows():
    t = row["label"][0]          # Strength->S etc.
    events.append({
        "year": int(row["year"]),
        "type": t,
        "mag" : round((row["count"]/max_cnt)**0.5 ,2),   # 0‒1 sqrt scale
        "x": quad_xy[row["label"]][0],
        "y": quad_xy[row["label"]][1],
        "rgb": color[row["label"]]
    })
    bullets.append({
        "year": int(row["year"]),
        "type": t,
        "facts":[{"text": row["summary"]}]
    })

# ── 4. write dashboard files ───────────────────────────────────────────────
events_path = DASHBOARD_DIR / "events.json"
bulletpoints_path = DASHBOARD_DIR / "bulletpoints.json"

events_path.write_text(json.dumps(events, indent=2))
bulletpoints_path.write_text(json.dumps(bullets, indent=2))

print(f"✓ wrote {events_path} and {bulletpoints_path}")
print(f"✓ Dashboard ready! Run 'cd dashboard && http-server .' to view")
