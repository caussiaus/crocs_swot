"""
Phase 3: Event Study & Sentiment Correlation
- Input: data/features/*_swot_llm.jsonl, stock prices (via yfinance)
- Output: data/features/*_event_study.parquet

Steps:
  1. Load event dates (10-K/Q filing dates)
  2. Compute abnormal returns around each date via market model
  3. Map each event’s sentiment (+/-) to CAR (cumulative abnormal return)
"""
# TODO: import pandas, statsmodels, yfinance

def run_event_study():
    pass

if __name__ == '__main__':
    run_event_study()
