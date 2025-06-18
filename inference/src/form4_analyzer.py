"""
Phase 4: Insider Trading Anomaly Detection (Form 4)
- Input: data/raw/*_4.jsonl
- Output: data/features/*_insider_activity.parquet

Steps:
  1. Parse Form 4 transactions (A vs D)
  2. Aggregate buys vs sells by period
  3. Fit log-normal model to detect abnormal selling
  4. Emit flags & volumes per quarter
"""
# TODO: import pandas, json

def analyze_form4():
    pass

if __name__ == '__main__':
    analyze_form4()
