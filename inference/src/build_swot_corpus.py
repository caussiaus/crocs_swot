#!/usr/bin/env python3
# DEPRECATED: Use B_yearly_swot_with_mistral.py for all SWOT corpus building logic.
"""yearly_swot_with_mistral.py – LOCAL‑ONLY v2

* Loads local **Mistral‑7B‑Instruct‑v0.3** once for BOTH embeddings
  (mean‑pooled hidden states) and generation.
* Uses Path objects + local_files_only=True so no call to HF Hub occurs.
* Switches to `AutoModelForCausalLM` (needed for generation).
* Minor: fixes YEAR_RE backslash.
"""
from __future__ import annotations
import os, re, json, itertools, logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import pandas as pd
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ── Local Mistral paths / config ────────────────────────────────────────────
MISTRAL_DIR: Path = (
    Path(__file__).resolve().parents[1] / "models" / "Mistral-7B-Instruct-v0.3"
).resolve()
assert (MISTRAL_DIR / "config.json").exists(), f"config.json not in {MISTRAL_DIR}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# ── File locations ──────────────────────────────────────────────────────────
CLUSTER_DIR = Path("processed/features_heading")
META_CSV    = Path("data/processed/metadata_report.csv")
OUT_JSON    = Path("yearly_swot.json")
OUT_CSV     = Path("yearly_swot.csv")

# ── 1. Load local model for BOTH embedding & generation ─────────────────────
print("Loading local Mistral‑7B …")

tokenizer = AutoTokenizer.from_pretrained(
    MISTRAL_DIR, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    MISTRAL_DIR,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)
model.eval()

@torch.no_grad()
def embed(sentences: List[str]) -> np.ndarray:
    """Mean‑pool last hidden state → unit‑norm embedding."""
    tok = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    out = model(**tok, output_hidden_states=True)
    last = out.hidden_states[-1]            # [B, L, H]
    emb  = last.mean(dim=1)                 # mean‑pool
    emb  = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

# generation pipeline for summaries
generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=100, do_sample=False,
)

def summarize(label: str, year: int, text: str) -> str:
    prompt = (
        f"You are an equity analyst. Create ONE concise bullet that summarises the company's {label.lower()} "
        f"based on the following sentences for fiscal {year}.\n\nSentences:\n{text}\n\nBullet:"
    )
    gen = generator(prompt, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    return gen.split("Bullet:")[-1].strip().replace("\n", " ")

# ── 2. Metadata look‑up ------------------------------------------------------
meta = pd.read_csv(META_CSV, dtype=str)
meta["accession"] = meta["filename"].str.extract(r"(^[^,]+)")
DATE = dict(zip(meta["accession"], meta["filing_date"]))
YEAR_RE = re.compile(r"(20\d{2}|19\d{2})")

# ── 3. Anchor embeddings for S/W/O/T ----------------------------------------
ANCHOR_TEXT = {
    "Strength":    "We hold a unique competitive advantage and strong brand loyalty.",
    "Weakness":    "Our operational inefficiencies and high costs hurt profitability.",
    "Opportunity": "We see significant market growth and expansion potential.",
    "Threat":      "Foreign exchange volatility and intense competition pose risks."
}
ANCHOR_EMB = embed(list(ANCHOR_TEXT.values()))
LABELS     = list(ANCHOR_TEXT.keys())

# ── 4. Collect sentences -----------------------------------------------------
rows: List[Dict] = []
for sub in ("10-K", "10-Q"):
    for fp in (CLUSTER_DIR / sub).glob("*.json"):
        accession = fp.stem.split("_")[0]
        date = DATE.get(accession)
        if not date:
            m = YEAR_RE.search(fp.stem)
            if m:
                date = f"{m.group(0)}-01-01"
        if not date:
            continue
        year = int(date[:4])
        data = json.loads(fp.read_text())
        if isinstance(data, dict) and "sentences" in data:
            sents = data["sentences"]
        else:
            sents = list(itertools.chain.from_iterable(c.get("examples",[]) for c in data))
        rows.extend({"year": year, "text": s} for s in sents)

if not rows:
    raise SystemExit("No sentences found — check paths!")

df = pd.DataFrame(rows)

# ── 5. Classify via nearest anchor -----------------------------------------
labels = []
BATCH = 64
for i in range(0, len(df), BATCH):
    embs = embed(df["text"].iloc[i:i+BATCH].tolist())
    idx  = (embs @ ANCHOR_EMB.T).argmax(axis=1)
    labels.extend([LABELS[j] for j in idx])

df["label"] = labels

# ── 6. Aggregate per year × label -------------------------------------------
agg = (
    df.groupby(["year", "label"])  # group
      .agg(cnt=("text", "size"), sample=("text", lambda s: " ".join(s.tolist()[:80])))
      .reset_index()
)
LATEST = df["year"].max()
agg = agg[agg["year"] >= LATEST - 14]   # last 15 fiscal years

# ── 7. Summarise -------------------------------------------------------------
print("Generating summaries … (this may take ~1‑2 min)")
agg["summary"] = agg.apply(lambda r: summarize(r["label"], int(r["year"]), r["sample"]), axis=1)

# ── 8. Save ------------------------------------------------------------------
report: Dict[int, Dict[str, Dict]] = defaultdict(dict)
for _, r in agg.iterrows():
    report[int(r["year"])][r["label"]] = {"count": int(r["cnt"]), "summary": r["summary"]}

OUT_JSON.write_text(json.dumps(report, indent=2))
agg.to_csv(OUT_CSV, index=False)
print(f"✓ wrote {OUT_JSON} and {OUT_CSV}")
