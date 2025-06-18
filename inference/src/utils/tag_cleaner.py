"""
File: src/utils/tag_cleaner.py

Utility to strip HTML/XBRL tags while preserving mapping from cleaned text indices
back to original raw text indices. This mapping allows downstream modules (e.g., KOR-based
feature extraction) to locate extracted data back in the original filings.
"""
import re
import html

class TagCleaner:
    def __init__(self, retain_map=True):
        """
        :param retain_map: if True, records a mapping list where mapping[i] = original_index
                           of the i-th character in the cleaned text.
        """
        self.retain_map = retain_map
        self.clean_text = ''
        self.mapping = []

    def clean(self, raw: str) -> str:
        """
        Strips all HTML/XML/XBRL tags, collapses whitespace, and returns the cleaned text.
        If retain_map=True, populates self.mapping such that:
          cleaned_text[i] originates from raw[self.mapping[i]].
        """
        clean_chars = []
        map_chars = []
        in_tag = False

        # Iterate through raw text, skipping tag content
        for orig_idx, ch in enumerate(raw):
            if ch == '<':
                in_tag = True
                continue
            if in_tag:
                if ch == '>':
                    in_tag = False
                continue
            # Outside tags: keep char
            if ch.isspace():
                ch = ' '
            clean_chars.append(ch)
            if self.retain_map:
                map_chars.append(orig_idx)

        # Collapse multiple spaces into one
        cleaned = re.sub(r' +', ' ', ''.join(clean_chars))

        # Build final mapping aligning with collapsed text
        if self.retain_map:
            final_map = []
            ci = 0
            prev_space = False
            for ch in cleaned:
                if ch == ' ':
                    if prev_space:
                        continue
                    prev_space = True
                else:
                    prev_space = False
                final_map.append(map_chars[ci])
                ci += 1
            self.mapping = final_map

        # Unescape HTML entities
        self.clean_text = html.unescape(cleaned)
        return self.clean_text

    def get_original_index(self, clean_index: int) -> int:
        """
        Returns the index in the original raw text corresponding to the given index
        in the cleaned text. If retain_map=False or index out of bounds, returns None.
        """
        if not self.retain_map or clean_index < 0 or clean_index >= len(self.mapping):
            return None
        return self.mapping[clean_index]

    def reset(self):
        """
        Clears stored clean_text and mapping, to reuse the instance for a new document.
        """
        self.clean_text = ''
        self.mapping = []

# Example usage:
# cleaner = TagCleaner(retain_map=True)
# clean = cleaner.clean(raw_html)
# orig_idx = cleaner.get_original_index(100)  # map cleaned pos 100 back to raw_text

if __name__ == "__main__":
    import os
    from pathlib import Path
    import warnings
    try:
        import chardet
    except ImportError:
        raise ImportError("Please install chardet: pip install chardet")

    RAW_BASE = Path(__file__).resolve().parents[2] / 'data' / 'raw'
    OUT_BASE = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'de-tagged'

    for sec_type in ['10-K', '10-Q']:
        in_dir = RAW_BASE / sec_type
        out_dir = OUT_BASE / sec_type
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in os.listdir(in_dir):
            in_path = in_dir / fname
            out_path = out_dir / fname.replace('.html', '.txt')
            if in_path.is_file():
                # Detect encoding
                with open(in_path, 'rb') as f:
                    raw_bytes = f.read()
                    result = chardet.detect(raw_bytes)
                    encoding = result['encoding'] or 'utf-8'
                try:
                    raw = raw_bytes.decode(encoding, errors='strict')
                except Exception:
                    try:
                        raw = raw_bytes.decode('latin-1', errors='replace')
                        warnings.warn(f"{in_path}: Decoded with latin-1 due to encoding issues.")
                    except Exception as e:
                        warnings.warn(f"Skipping {in_path}: Cannot decode as text. {e}")
                        continue
                cleaner = TagCleaner(retain_map=False)
                cleaned = cleaner.clean(raw)
                with open(out_path, 'w', encoding='utf-8') as wf:
                    wf.write(cleaned)
                print(f"Processed {in_path} -> {out_path}")
