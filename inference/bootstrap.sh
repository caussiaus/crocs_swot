#!/usr/bin/env bash
# crocs_data/inference/bootstrap.sh
# Fully scaffold the inference directory, create conda env, and outline folder structure.
set -euo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_NAME="crocs-inference"

# 1) Ensure conda in PATH
if ! command -v conda &> /dev/null; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
  else
    echo "❌ conda not found. Please install Miniconda first." >&2
    exit 1
  fi
fi

# 2) Create or update the conda environment
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "• Updating existing conda env '${ENV_NAME}'"
  conda env update -n "$ENV_NAME" -f "$HERE/env.yml" --prune
else
  echo "• Creating new conda env '${ENV_NAME}'"
  conda env create -f "$HERE/env.yml"
fi

# 3) Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 4) Install pip-only dependencies
echo "• Installing pip dependencies"
pip install --upgrade pip
pip install -r "$HERE/post_pip.txt"

# 5) Scaffold directory structure
echo "• Scaffolding inference folders and placeholder scripts"
# Core directories
dirs=(
  "$HERE/src"
  "$HERE/src/utils"
  "$HERE/data/raw"
  "$HERE/data/processed"
  "$HERE/data/features"
)
for d in "${dirs[@]}"; do
  mkdir -p "$d"
done

# Placeholder files with comments
cat > "$HERE/src/utils/tag_cleaner.py" << 'EOF'
"""
Utility to strip HTML/XBRL tags while preserving mapping to original offsets.
"""
import re

class TagCleaner:
    def __init__(self, retain_map=True):
        self.retain_map = retain_map
        self.clean_text = ''
        self.mapping = []

    def clean(self, raw: str) -> str:
        clean_chars, map_chars = [], []
        in_tag = False
        for i, ch in enumerate(raw):
            if ch == '<': in_tag = True; continue
            if in_tag:
                if ch == '>': in_tag = False
                continue
            if ch.isspace(): ch = ' '
            clean_chars.append(ch)
            if self.retain_map: map_chars.append(i)
        cleaned = re.sub(r' +', ' ', ''.join(clean_chars))
        # mapping logic omitted for brevity
        self.clean_text = cleaned
        return cleaned

    def get_original_index(self, idx: int) -> int:
        return self.mapping[idx] if self.retain_map and idx < len(self.mapping) else idx
EOF

cat > "$HERE/src/feature_extraction_kor.py" << 'EOF'
"""
Feature extractor scaffold: locate earnings, EPS, MD&A, etc. Use TagCleaner.
"""
import json
from pathlib import Path
import re
from utils.tag_cleaner import TagCleaner

INPUT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'processed'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'features'
OUTPUT_DIR.mkdir(exist_ok=True)

# Define regex patterns here
PATTERNS = {...}
HEADERS = {...}

def extract_features():
    cleaner = TagCleaner(retain_map=True)
    # iterate JSONL chunks
    # for each chunk: cleaner.clean(), find patterns, write JSONL

if __name__ == '__main__':
    extract_features()
EOF

cat > "$HERE/src/swot_extract.py" << 'EOF'
"""
SWOT extractor scaffold: LLM inference using HF token
"""
import os
from pathlib import Path
from dotenv import load_dotenv
# import transformers, pipeline, TagCleaner...

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
# Load LLM model, set up pipeline

def analyze_chunks():
    # read processed chunks, call LLM, output SWOT JSONL
    pass

if __name__ == '__main__':
    analyze_chunks()
EOF

# 6) GPU sanity check
echo "• Verifying GPU setup"
python - << 'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available(): print('Device:', torch.cuda.get_device_name(0))
PY

echo "✅ Inference scaffold complete."

# 7) Final usage hint
echo
echo "Next steps:"
echo "  cd $HERE && conda activate $ENV_NAME"
echo "  # draft or fill in scripts under src/"
echo "  python -m src.feature_extraction_kor   # test KOR scaffold"
echo "  python -m src.swot_extract             # test SWOT scaffold"
