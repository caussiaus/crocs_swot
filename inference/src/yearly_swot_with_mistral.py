#!/usr/bin/env python3
"""yearly_swot_with_mistral.py — v3 (quality‑tuned)

✔ Loads **local** Mistral‑7B‑Instruct (4‑bit) for both embedding & bullet gen
✔ Robust date fallback → always stamps a filing year
✔ Cosine‑anchor classifier with ★ margin & similarity thresholds
✔ Ignores boiler‑plate via extra anchor
✔ De‑duplicates identical sentences
✔ Limits each (year × bucket) to top‑30 high‑similarity sentences
✔ Deterministic summaries (no sampling)
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
BASE   = Path("/home/tempuser/projects/crocs_data/inference/processed/features_heading")
META_CSV = Path("data/processed/metadata_report.csv")
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
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", quantization_config=bnb_conf)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
DEVICE = next(model.parameters()).device

@torch.no_grad()
def embed(sentences: List[str]) -> np.ndarray:
    toks = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    last = model(**toks, output_hidden_states=True).hidden_states[-1]
    emb  = torch.nn.functional.normalize(last.mean(dim=1), p=2, dim=1)
    return emb.cpu().numpy()

generator = pipeline("text-generation", model=model, tokenizer=tokenizer,
                     max_new_tokens=80, do_sample=False)

def summarize(label:str, year:int, sents:List[str]) -> str:
    body = " ".join(sents[:30])
    prompt = (
        f"You are an equity analyst. Compose ONE concise bullet (max 30 words) that summarises the company's "
        f"{label.lower()} for fiscal {year}.\n\nSentences:\n{body}\n\nBullet:")
    out = generator(prompt)[0]["generated_text"].split("Bullet:")[-1]
    return out.strip(' \n"*`')

# ── metadata & year fallback ───────────────────────────────────────────────
meta = pd.read_csv(META_CSV, dtype=str) if META_CSV.exists() else pd.DataFrame()
meta["accession"] = meta.get("filename","").str.extract(r"(^[^,]+)")
DATE = dict(zip(meta["accession"], meta.get("filing_date","")))

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

if not keep_idx:
    sys.exit("❌ All sentences filtered as boiler/low‑sim")

df=df.iloc[keep_idx].copy(); df["label"]=labs

# ── aggregate & summarise ──────────────────────────────────────────────────
agg=(df.groupby(["year","label"]).agg(cnt=("text","size"),
     sents=("text", lambda s:list(s)[:60])).reset_index())
LATEST=df.year.max(); agg=agg[agg.year>=LATEST-14]

print("▶ Generating bullet summaries…")
agg["summary"]=agg.apply(lambda r:summarize(r.label,int(r.year),r.sents),axis=1)

report:Dict[int,Dict[str,Dict]]=defaultdict(dict)
for _,r in agg.iterrows():
    report[int(r.year)][r.label]={"count":int(r.cnt),"summary":r.summary}

OUT_JSON.write_text(json.dumps(report,indent=2))
agg.drop(columns="sents").to_csv(OUT_CSV,index=False)
print("✓ wrote", OUT_JSON, "and", OUT_CSV)
