#!/usr/bin/env python3
#this was the one that made it easier to extract sentences adn key points.
"""A_heading_preserving_converter.py
Convert raw EDGAR 10-K / 10-Q HTML filings into *heading-preserving* plain-text.

•   Keeps <h1> … <h6> tags as Markdown (#, ##, ### …).
•   Detects SEC "ITEM X." headings that are *not* wrapped in <h*> tags and
    promotes them to heading level 2.
•   Outputs UTF-8 text where headings and sub-headings remain in order – ideal
    for the chunking / embedding workflow shown in your screenshot.
•   Tracks metadata (filing dates, report types, accession numbers) for downstream analysis.

Usage
-----
Single file → stdout:
    python A_heading_preserving_converter.py path/to/10k.htm

Batch convert directory (writes alongside original, .txt extension):
    python A_heading_preserving_converter.py path/to/raw_dir -o path/to/txt_dir -r

Dependencies
------------
    pip install beautifulsoup4 html2text tqdm
"""
from __future__ import annotations
import re, sys, argparse, html, json
from pathlib import Path
from typing import Iterable
from bs4 import BeautifulSoup, NavigableString, Tag
from html2text import HTML2Text
from tqdm import tqdm
from datetime import datetime

# ───────────────────────────── HTML → Markdown ───────────────────────────────
CONVERTER = HTML2Text()
CONVERTER.body_width = 0        # no hard-wrap
CONVERTER.ignore_links = True
CONVERTER.ignore_images = True
CONVERTER.ignore_tables = True
CONVERTER.single_line_break = True

ITEM_RE = re.compile(r"^\s*ITEM\s+([0-9]+[A-Z]?)\.?\s*(.*)$", re.I)

HEADING_LEVEL_FROM_ITEM = 2     # markdown level (##) for ITEM headings

def extract_metadata(file_path: Path) -> dict:
    """Extract metadata from file path and content."""
    # Extract report type from directory structure
    report_type = file_path.parent.name  # '10-K' or '10-Q'
    
    # Extract accession number and filing date from filename
    filename = file_path.name
    accession_match = re.match(r'(\d{10}-\d{2}-\d{6,})', filename)
    accession_number = accession_match.group(1) if accession_match else None
    
    # Extract filing date from filename
    date_match = re.search(r'(\d{4})[-_]?([01]\d)[-_]?([0-3]\d)', filename)
    if date_match:
        filing_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    else:
        # Fallback to file modification date
        try:
            ts = file_path.stat().st_mtime
            filing_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        except Exception:
            filing_date = None
    
    # Extract CIK from accession number
    cik = accession_number.split('-')[0] if accession_number else None
    
    return {
        'filename': filename,
        'accession_number': accession_number,
        'report_type': report_type,
        'filing_date': filing_date,
        'cik': cik,
        'file_path': str(file_path)
    }

def extract_text(html_path: Path) -> tuple[str, dict]:
    """Read HTML, keep headings, return markdown-like plain text and metadata."""
    # Extract metadata first
    metadata = extract_metadata(html_path)
    
    soup = BeautifulSoup(html_path.read_text("utf8", errors="ignore"), "lxml")

    # Promote <b> or <p class=...> that look like ITEM headings → <h2>
    for tag in soup.find_all(text=ITEM_RE):
        parent: Tag | None = tag.parent
        if not parent or parent.name.startswith("h"):
            continue
        m = ITEM_RE.match(tag)
        # wrap in new <h2>
        new = soup.new_tag("h2")
        new.string = m.group(0)
        parent.replace_with(new)

    # Use html2text to convert, headings preserved
    md = CONVERTER.handle(str(soup))

    # Post-process: collapse >2 blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    # Trim trailing whitespace
    md = "\n".join(line.rstrip() for line in md.splitlines())

    return md.strip() + "\n", metadata

# ───────────────────────────── CLI glue ──────────────────────────────────────

def iter_files(path: Path, recurse: bool) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        glob = path.rglob if recurse else path.glob
        for p in glob("*.htm*"):
            yield p


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Convert EDGAR HTML to heading-preserving text")
    ap.add_argument("input", type=Path, help="HTML file or directory")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output dir (defaults next to input)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into sub-dirs when input is a dir")
    args = ap.parse_args(argv)

    files = list(iter_files(args.input, args.recursive))
    if not files:
        sys.exit("No .htm files found")

    for fp in tqdm(files, desc="Converting"):
        text, metadata = extract_text(fp)
        if args.out:
            out_dir = args.out / fp.relative_to(args.input).parent if args.input.is_dir() else args.out
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (fp.stem + ".txt")
        else:
            out_path = fp.with_suffix(".txt")
        out_path.write_text(text, encoding="utf8")

def batch_process_raw_to_processed():
    """Process all 10-K and 10-Q files from data/raw to data/processed/heading_preserved."""
    base_in = Path(__file__).parent.parent.parent / "data" / "raw"
    base_out = Path(__file__).parent.parent.parent / "data" / "processed" / "heading_preserved"
    
    # Track metadata for all processed files
    all_metadata = []
    
    for form_type in ["10-K", "10-Q"]:
        in_dir = base_in / form_type
        out_dir = base_out / form_type
        if not in_dir.exists():
            continue
        files = list(in_dir.rglob("*.htm*"))
        if not files:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {len(files)} {form_type} files...")
        for file_path in tqdm(files, desc=f"Converting {form_type}"):
            try:
                text, metadata = extract_text(file_path)
                out_path = out_dir / f"{file_path.stem}.txt"
                out_path.write_text(text, encoding="utf8")
                
                # Store metadata for this file
                metadata['output_path'] = str(out_path)
                all_metadata.append(metadata)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Save metadata summary
    metadata_path = base_out / "processing_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"✓ Processed {len(all_metadata)} files")
    print(f"✓ Metadata saved to {metadata_path}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No arguments provided. Running batch mode for data/raw/10-K and 10-Q...")
        batch_process_raw_to_processed()
    else:
        main()