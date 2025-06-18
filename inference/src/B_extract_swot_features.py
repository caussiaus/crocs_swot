#!/usr/bin/env python3
#THIS WAS THE ONE THAT WORKED FOR THE SENTENCE EXTRACTION
"""heading_swot_extractor.py  – fixed

Crucial fixes
-------------
1. **Heading gate logic** – now sets a boolean `keep_context` once per heading so
   narrative lines under *Item 1, 1A, 2, 7, 7A* are processed; all other Items
   are skipped cleanly. Previous version skipped *everything* because
   `stack[0][0].startswith(k)` check still included the trailing description.
2. Simpler main loop: we `continue` immediately after handling a heading so the
   heading line itself isn't tokenised.
3. Added support for bullet/dash list lines that belong to the kept context.
4. Mild refactor & type hints – behaviour unchanged elsewhere.
"""
from __future__ import annotations
import os, re, json, math, pickle, uuid, logging, warnings
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import hdbscan
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None
    warnings.warn("faiss not installed – duplicate suppression disabled", RuntimeWarning)

# ─── paths ──────────────────────────────────────────────────────────────────
BASE = Path("data/processed/heading_preserved")
OUT_BASE = BASE  # If output is still needed, write to the same heading_preserved folder
SUBDIRS = ["10-K", "10-Q"]

# ─── heading config ─────────────────────────────────────────────────────────
KEEP_TOP = {"Item 1.", "Item 1A.", "Item 2.", "Item 7.", "Item 7A."}
ITEM_PREFIX_RE = re.compile(r"item\s+\d+[a-z]?\.", re.I)        # captures "Item 1." etc.
ITEM_LINE_RE   = re.compile(r"^\s*(ITEM\s+\d+[A-Z]?\.)", re.I) # lines that are pure ITEM headings
MD_H_RE        = re.compile(r"^(#+)\s*(.+)")                     # markdown headings

# ─── filters ───────────────────────────────────────────────────────────────
SWOT_KWS = [
    "brand","reputation","competitive advantage","innovation","patent","proprietary",
    "margin","liquidity","supply chain","cybersecurity","volatility","inflation",
    "debt","growth","e-commerce","expansion","acquisition","sustainability","esg",
    "recall","foreign currency","strategy","automation","analytics","ai"
]
KW_RE      = re.compile(r"\b("+"|".join(re.escape(k) for k in SWOT_KWS)+r")\b", re.I)
BOILER_RE  = re.compile(r"(?xi)(forward\s+looking\s+statement|sarbanes\-oxley|table\s+of\s+contents)")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# ─── embedder ───────────────────────────────────────────────────────────────
DEVICE = 0 if torch.cuda.is_available() else -1
EMB_MODEL = os.getenv("SW_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer(EMB_MODEL, device="cuda" if DEVICE == 0 else "cpu")
EMB_DIM = embedder.get_sentence_embedding_dimension()

# ─── TF‑IDF rarity ─────────────────────────────────────────────────────────
IDF_CACHE = BASE.parent / "idf_heading.pkl"
if IDF_CACHE.exists():
    VEC, IDF_MAP = pickle.load(IDF_CACHE.open("rb"))
else:
    corpus: List[str] = []
    for sd in SUBDIRS:
        for fp in (BASE/sd).glob("*.txt"):
            corpus.extend(SENT_SPLIT.split(fp.read_text("utf8", errors="ignore")))
    VEC = TfidfVectorizer(min_df=2, max_df=0.95, stop_words="english").fit(corpus)
    IDF_MAP = dict(zip(VEC.get_feature_names_out(), VEC.idf_))
    pickle.dump((VEC, IDF_MAP), IDF_CACHE.open("wb"))
IDF_CUT = float(np.quantile(list(IDF_MAP.values()), 0.01))

def avg_idf(text: str) -> float:
    toks = re.findall(r"[A-Za-z]+", text.lower())
    return float(np.mean([IDF_MAP.get(t, max(IDF_MAP.values())) for t in toks])) if toks else max(IDF_MAP.values())

# ─── FAISS dedupe ──────────────────────────────────────────────────────────
class Deduper:
    def __init__(self):
        self.enabled = faiss is not None
        self.index   = faiss.IndexFlatIP(EMB_DIM) if self.enabled else None
        self.ids: List[str] = []
    def seen(self, emb: np.ndarray, thr: float=0.94) -> bool:
        if not self.enabled or not self.ids:
            return False
        D,_ = self.index.search(normalize(emb.reshape(1,-1)).astype("float32"), k=3)
        return (D > thr).any()
    def add(self, emb: np.ndarray):
        if self.enabled:
            self.index.add(normalize(emb.reshape(1,-1)).astype("float32"))
            self.ids.append(uuid.uuid4().hex)

deduper = Deduper()

# ─── helpers ───────────────────────────────────────────────────────────────
logging.basicConfig(format="%(levelname)7s | %(message)s", level=logging.INFO)
log = logging.getLogger("heading-swot")

def sentences(line: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT.split(line) if len(s.strip()) > 20]

# ─── core function ─────────────────────────────────────────────────────────

def process_file(fp: Path, out_dir: Path) -> None:
    lines = fp.read_text("utf8", errors="ignore").splitlines()

    stack: List[Tuple[str,int]] = []   # (heading, depth)
    keep_context = False               # are we inside a wanted Item?
    candidates: List[str] = []

    for ln in lines:
        # 1) heading detection ------------------------------------------------
        md = MD_H_RE.match(ln)
        itm = ITEM_LINE_RE.match(ln)
        if md or itm:
            # figure heading str & depth
            if md:
                depth = len(md.group(1))
                text  = md.group(2).strip()
            else:  # item heading not markdown
                depth = 2
                text  = itm.group(1).strip()
            # maintain stack
            while stack and stack[-1][1] >= depth:
                stack.pop()
            stack.append((text, depth))
            # update keep_context flag based on *top* heading
            top_match = ITEM_PREFIX_RE.match(stack[0][0])
            keep_context = top_match and top_match.group(0).title() in KEEP_TOP
            continue  # do not tokenise the heading line itself

        # skip lines outside desired sections
        if not keep_context:
            continue

        # 2) narrative sentence extraction -----------------------------------
        for sent in sentences(ln):
            if KW_RE.search(sent) and not BOILER_RE.search(sent):
                full_path = " › ".join(h for h,_ in stack)
                candidates.append(f"{full_path} :: {sent}")
        # also keep bullet / dash list lines if they mention keywords
        if ln.lstrip().startswith(("•","-","*")) and KW_RE.search(ln):
            full_path = " › ".join(h for h,_ in stack)
            candidates.append(f"{full_path} :: {ln.lstrip('*•- ').strip()}")

    if not candidates:
        return

    # 3) rarity + dedupe + embed --------------------------------------------
    kept: List[Tuple[str,np.ndarray]] = []
    for s in candidates:
        plain = s.split(" :: ",1)[1]
        if avg_idf(plain) < IDF_CUT:
            continue
        emb = embedder.encode(s, show_progress_bar=False)
        if deduper.seen(emb):
            continue
        deduper.add(emb)
        kept.append((s,emb))
    if not kept:
        return

    # 4) cluster by top heading----------------------------------------------
    by_head: Dict[str, List[Tuple[str,np.ndarray]]] = defaultdict(list)
    for s,e in kept:
        top = s.split(" › ",1)[0]
        by_head[top].append((s,e))

    results = []
    for top, pairs in by_head.items():
        if len(pairs) < 2:
            continue
        vecs = np.vstack([e for _,e in pairs])
        labels = hdbscan.HDBSCAN(min_cluster_size=max(2,len(pairs)//15)).fit_predict(vecs)
        clus: Dict[int,List[str]] = defaultdict(list)
        for (s,_),lbl in zip(pairs,labels):
            clus[lbl].append(s)
        for cid,slist in clus.items():
            if cid==-1 or len(slist)<2:
                continue
            results.append({
                "heading": top,
                "cluster": int(cid),
                "examples": [txt.split(" :: ",1)[1] for txt in slist[:8]]
            })

    # 5) write ----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    if results:
        (out_dir/f"{fp.stem}_clusters.json").write_text(json.dumps(results,indent=2),"utf8")
    else:
        fallback=[txt.split(" :: ",1)[1] for txt,_ in kept][:40]
        (out_dir/f"{fp.stem}_fallback.json").write_text(json.dumps({"sentences":fallback},indent=2),"utf8")

# ─── main walker ───────────────────────────────────────────────────────────

def main():
    for sd in SUBDIRS:
        in_dir  = BASE / sd
        out_dir = OUT_BASE / sd
        out_dir.mkdir(parents=True, exist_ok=True)
        for fp in tqdm(sorted(in_dir.glob("*.txt")), desc=sd):
            try:
                process_file(fp, out_dir)
            except Exception as e:
                log.error(f"{fp.name}: {e}")

if __name__ == "__main__":
    main()
