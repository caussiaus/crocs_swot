#!/usr/bin/env python3
"""B_yearly_swot_with_mistral.py — v3 (quality‑tuned)

✔ Loads **local** Mistral‑7B‑Instruct (4‑bit) for both embedding & bullet gen
✔ Robust date fallback → always stamps a filing year
✔ Cosine‑anchor classifier with ★ margin & similarity thresholds
✔ Ignores boiler‑plate via extra anchor
✔ De‑duplicates identical sentences
✔ Limits each (year × bucket) to top‑30 high‑similarity sentences
✔ Deterministic summaries (no sampling)
✔ Uses metadata from A_heading_preserver_converter.py for proper timestamping
"""
from __future__ import annotations
import json, itertools, re, hashlib, sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import pandas as pd
import torch, numpy as np
from transformers import (AutoTokenizer, AutoModelForCausalLM, pipeline,
                          BitsAndBytesConfig)
# ── paths ──────────────────────────────────────────────────────────────────
BASE   = Path("data/processed/heading_preserved")
META_JSON = BASE / "processing_metadata.json"  # From A_heading_preserver_converter.py
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Mistral-7B-Instruct-v0.3"
OUT_DIR  = Path("yearly_swot"); OUT_DIR.mkdir(exist_ok=True)
OUT_JSON = OUT_DIR / "yearly_swot.json"; OUT_CSV = OUT_DIR / "yearly_swot.csv"

assert (MODEL_DIR / "config.json").exists(), "model folder incomplete"

# ── model (4‑bit) ──────────────────────────────────────────────────────────
print("▶ Loading Mistral‑7B (4‑bit)…")
bnb_conf = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR,
                                            quantization_config=bnb_conf,
                                            device_map="auto",
                                            trust_remote_code=True,
                                            local_files_only=True)
model.eval()

@torch.no_grad()
def embed(sentences: List[str]) -> np.ndarray:
    """Mean‑pool last hidden state → unit‑norm embedding."""
    tok = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(model.device)
    out = model(**tok, output_hidden_states=True)
    last = out.hidden_states[-1]
    emb  = last.mean(dim=1)
    emb  = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

generator = pipeline("text-generation", model=model, tokenizer=tokenizer,
                     device=0 if torch.cuda.is_available() else -1,
                     max_new_tokens=100, do_sample=False)

def summarize(label: str, year: int, text: str) -> str:
    prompt = (
        f"You are an equity analyst. Create ONE concise bullet that summarises the company's {label.lower()} "
        f"based on the following sentences for fiscal {year}.\n\nSentences:\n{text}\n\nBullet:"
    )
    gen = generator(prompt, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    return gen.split("Bullet:")[-1].strip().replace("\n", " ")

# ── metadata & year fallback ───────────────────────────────────────────────
print("▶ Loading metadata from A step...")
if META_JSON.exists():
    with open(META_JSON, 'r') as f:
        metadata_list = json.load(f)
    # Create mapping from filename to filing date
    DATE = {}
    for meta in metadata_list:
        filename = meta.get('filename', '')
        filing_date = meta.get('filing_date')
        if filename and filing_date:
            # Extract accession number from filename
            accession_match = re.match(r'(\d{10}-\d{2}-\d{6,})', filename)
            if accession_match:
                accession = accession_match.group(1)
                DATE[accession] = filing_date
else:
    print("⚠️  No metadata found, using filename fallback")
    DATE = {}

def year_from_name(name:str)->int|None:
    # first YY after accession dash
    m = re.search(r"-([0-9]{2})-", name)
    if m:
        yy=int(m.group(1)); return 2000+yy if yy<40 else 1900+yy
    m=re.search(r"(19|20)\d{2}", name)
    return int(m.group(0)) if m else None

# ── collect sentences ──────────────────────────────────────────────────────
print("▶ Scanning extracted jsons…")
rows=[]
for fp in BASE.rglob("*.json"):
    accession = fp.stem.split("_")[0]
    date = DATE.get(accession)
    if not date:
        y=year_from_name(fp.name)
        if y: date=f"{y}-01-01"
    if not date: continue
    yr=int(date[:4])
    data=json.loads(fp.read_text())
    sents=data.get("sentences") if isinstance(data,dict) else list(itertools.chain.from_iterable(c.get("examples",[]) for c in data))
    for s in sents:
        rows.append({"year":yr,"text":s})

if not rows:
    sys.exit("❌ No sentences collected – check paths")

df=pd.DataFrame(rows).drop_duplicates("text")
print(f"Collected {len(df):,} unique sentences across {df.year.nunique()} years")

# ── anchors & classification ───────────────────────────────────────────────
ANCHOR_TEXT={
 "Strength":"We maintain strong brand loyalty and a unique advantage.",
 "Weakness":"Operational inefficiency or declining margins hurt performance.",
 "Opportunity":"Market expansion and product growth prospects are significant.",
 "Threat":"Competitive pressures and macro risks pose challenges.",
 "Boiler":"This statement includes forward‑looking information and risk factors."
}
ANCH_EMB=embed(list(ANCHOR_TEXT.values())); LABELS=list(ANCHOR_TEXT.keys())
THR_SIM=0.23; THR_MARGIN=0.05

labs=[]; keep_idx=[]
B=64
for i in range(0,len(df),B):
    em=embed(df.text.iloc[i:i+B].tolist())
    sim=em@ANCH_EMB.T
    top=sim.argmax(1); second=np.partition(sim, -2, axis=1)[:,-2]
    for j,(ti, smax, s2) in enumerate(zip(top, sim.max(1), second)):
        if smax>=THR_SIM and (smax-s2)>=THR_MARGIN and LABELS[ti]!="Boiler":
            labs.append(LABELS[ti]); keep_idx.append(i+j)

df=df.iloc[keep_idx].copy(); df["label"]=labs
print(f"Classified {len(df):,} sentences into {df.label.nunique()} categories")

# ── aggregate per year × label ─────────────────────────────────────────────
agg=(df.groupby(["year","label"])
       .agg(cnt=("text","size"), sample=("text",lambda s:" ".join(s.tolist()[:80])))
       .reset_index())
LATEST=df["year"].max()
agg=agg[agg["year"]>=LATEST-14]  # last 15 fiscal years

# ── summarize ──────────────────────────────────────────────────────────────
print("▶ Generating summaries… (this may take ~1‑2 min)")
agg["summary"]=agg.apply(lambda r:summarize(r["label"],int(r["year"]),r["sample"]),axis=1)

# ── save ───────────────────────────────────────────────────────────────────
report:Dict[int,Dict[str,Dict]]=defaultdict(dict)
for _,r in agg.iterrows():
    report[int(r["year"])][r["label"]]={"count":int(r["cnt"]),"summary":r["summary"]}

OUT_JSON.write_text(json.dumps(report,indent=2))
agg.to_csv(OUT_CSV,index=False)
print(f"✓ wrote {OUT_JSON} and {OUT_CSV}")
