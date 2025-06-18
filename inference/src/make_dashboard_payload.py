#!/usr/bin/env python3
"""
make_dashboard_payload.py
─────────────────────────
Generate events.json + bulletpoints.json for the SWOT quadrant
dashboard, using an LLM summarizer to create human-readable
bullet points per year.
"""

from __future__ import annotations
import json, itertools, textwrap
from pathlib import Path
import pandas as pd
from transformers import pipeline

DATA   = Path("swot_sentences.parquet")          # produced earlier
DEST   = Path("dist");  DEST.mkdir(exist_ok=True)

# ── 1. load sentences ───────────────────────────────────────────────────────
if DATA.suffix == ".parquet":
    df = pd.read_parquet(DATA)
else:                                            # fallback to CSV
    df = pd.read_csv(DATA)

# ── 2. basic stats per bucket ───────────────────────────────────────────────
bucket = (df.groupby(["year","label"])
            .agg(cnt=("text","size"),
                 sample=("text", lambda s: " ".join(s.head(40))))
            .reset_index())

max_cnt = bucket["cnt"].max()

# ── 3. Summarizer (Bart CNN) ────────────────────────────────────────────────
summarizer = pipeline("summarization",
                      model="facebook/bart-large-cnn",
                      tokenizer="facebook/bart-large-cnn",
                      device=0 if pd.get_option("display.max_rows") else -1)

def summarize(text:str)->str:
    short = textwrap.shorten(text, 600, placeholder=" … ")
    return summarizer(short, max_length=45, min_length=15, do_sample=False)[0]["summary_text"]

bucket["summary"] = bucket["sample"].map(summarize)

# ── 4. map into dashboard schema ────────────────────────────────────────────
quad_xy = {"S":(.25,.25), "W":(.75,.25), "O":(.25,.75), "T":(.75,.75)}
color   = {"S":"46,125,50", "O":"46,125,50", "W":"229,57,53", "T":"229,57,53"}

events  = []
bullets = []

for _, row in bucket.iterrows():
    t = row["label"][0]          # Strength->S etc.
    events.append({
        "year": int(row["year"]),
        "type": t,
        "mag" : round((row["cnt"]/max_cnt)**0.5 ,2),   # 0‒1 sqrt scale
        "x": quad_xy[t][0],
        "y": quad_xy[t][1],
        "rgb": color[t]
    })
    bullets.append({
        "year": int(row["year"]),
        "type": t,
        "facts":[{"text": row["summary"]}]
    })

# ── 5. write ----------------------------------------------------------------
(Path(DEST/"events.json")
     .write_text(json.dumps(events, indent=2)))
(Path(DEST/"bulletpoints.json")
     .write_text(json.dumps(bullets, indent=2)))

print("✓ wrote dist/events.json and dist/bulletpoints.json")
