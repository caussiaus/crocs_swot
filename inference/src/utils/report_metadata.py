"""
File: src/utils/report_metadata.py

Utility to extract and attach metadata (report type, filing date) to each processed feature record.

Assumptions:
- Raw filings are stored under data/raw/<FORM>/ with filenames containing dates.
- Processed chunks are derived from raw filenames: chunk_id includes the raw filename prefix.

Functions:
  - get_report_metadata(raw_path: Path) -> dict with 'report_type', 'filing_date'
  - annotate_features(feature_jsonl: Path, output_jsonl: Path) -> writes enriched records

Usage:
  from pathlib import Path
  feature_file = Path('data/features/2024-Q1_10-Q_kor_features.jsonl')
  annotate_features(feature_file, feature_file.with_name(feature_file.stem + '_meta.jsonl'))
"""
import re
from pathlib import Path
from datetime import datetime

def get_report_metadata(raw_path: Path) -> dict:
    """
    Derive metadata from raw filing path.
    :param raw_path: Path to raw filing file (e.g., data/raw/10-K/0001334036-24-000001.html)
    :return: { 'report_type': '10-K', 'filing_date': '2024-03-01' }
    """
    report_type = raw_path.parent.name
    # Attempt to parse date from filename using YYYYMMDD or YYYY-MM-DD
    name = raw_path.stem
    # Common patterns
    m = re.search(r"(\d{4})[-_]?([01]\d)[-_]?([0-3]\d)", name)
    if m:
        filing_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    else:
        # Fallback: file modification date
        try:
            ts = raw_path.stat().st_mtime
            filing_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        except Exception:
            filing_date = None
    return { 'report_type': report_type, 'filing_date': filing_date }


def annotate_features(feature_jsonl: Path, output_jsonl: Path):
    """
    Read feature records (JSONL), attach metadata for each record based on its chunk origin.
    Writes annotated JSONL to output_jsonl.
    """
    import json
    from src.utils.tag_cleaner import TagCleaner

    # Build mapping: chunk_id -> raw file path
    # Assumes chunk_id format: '<raw_filename>::<chunk_index>'
    chunk_to_raw = {}
    for raw_form in Path(feature_jsonl.parents[2] / 'raw').iterdir():
        for raw_file in raw_form.glob('*'):
            key = raw_file.name  # without extension
            chunk_to_raw[key] = raw_file

    cleaner = TagCleaner(retain_map=False)
    with open(feature_jsonl, 'r', encoding='utf-8') as rf, \
         open(output_jsonl, 'w', encoding='utf-8') as wf:
        for line in rf:
            rec = json.loads(line)
            chunk_id = rec.get('chunk_id', '')
            raw_key = chunk_id.split('::')[0]
            raw_path = chunk_to_raw.get(raw_key)
            if raw_path:
                meta = get_report_metadata(raw_path)
            else:
                meta = { 'report_type': None, 'filing_date': None }
            rec['report_type'] = meta['report_type']
            rec['filing_date'] = meta['filing_date']
            wf.write(json.dumps(rec) + '\n')

# Example usage:
# annotate_features(Path('data/features/2024Q1_10-Q_kor_features.jsonl'),
#                   Path('data/features/2024Q1_10-Q_kor_features_meta.jsonl'))

if __name__ == "__main__":
    import csv
    import re
    from pathlib import Path
    import pandas as pd

    RAW_BASE = Path(__file__).resolve().parents[2] / 'data' / 'raw'
    LINKS_CSV = Path(__file__).resolve().parents[3] / 'edgar-scraper' / 'links' / 'CIK_0001334036_filings.csv'
    OUT_CSV = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'metadata_report.csv'

    # Load links CSV into a dict by accession number
    links_df = pd.read_csv(LINKS_CSV, dtype=str)
    links_df['accessionNumber'] = links_df['accessionNumber'].astype(str)
    links_lookup = {row['accessionNumber']: row for _, row in links_df.iterrows()}

    rows = []
    for form_folder in RAW_BASE.iterdir():
        if not form_folder.is_dir():
            continue
        folder_form = form_folder.name
        for f in form_folder.glob('*'):
            if not f.is_file():
                continue
            filename = f.name
            # Accession number: usually first part of filename
            m = re.match(r'(\d{10}-\d{2}-\d{6,})', filename)
            accession_number = m.group(1) if m else None
            cik = accession_number.split('-')[0] if accession_number else None
            # Filing date: try to extract from filename
            date_match = re.search(r'(\d{4})[-_]?([01]\d)[-_]?([0-3]\d)', filename)
            filing_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else None
            # Cross-check with links CSV
            csv_form = None
            url = None
            if accession_number and accession_number in links_lookup:
                csv_form = links_lookup[accession_number]['form']
                url = links_lookup[accession_number]['url']
            detected_form = folder_form
            mismatch_flag = (csv_form is not None and csv_form != folder_form)
            notes = ''
            if mismatch_flag:
                notes = f"Folder form {folder_form} != CSV form {csv_form}"
                print(f"WARNING: {filename}: {notes}")
            rows.append({
                'filename': filename,
                'accession_number': accession_number,
                'detected_form': detected_form,
                'folder_form': folder_form,
                'csv_form': csv_form,
                'filing_date': filing_date,
                'cik': cik,
                'url': url,
                'mismatch_flag': mismatch_flag,
                'notes': notes
            })
    # Write to CSV
    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Metadata report written to {OUT_CSV}")
