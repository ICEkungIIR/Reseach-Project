"""
data_prep.py

Purpose:
    Load raw Thai datasets (Wisesight Sentiment, Thai NER), clean them,
    and produce fixed train/val/test splits (single fixed seed, reused
    across every experiment config so split variance doesn't mix with
    rank variance).

Planned functions:
    - load_raw(task: str) -> pandas.DataFrame
    - clean_text(df) -> pandas.DataFrame
    - split_dataset(df, seed: int) -> (train, val, test)
    - save_processed(...) -> writes to data/processed/

Not yet implemented — scaffold only.
"""

def load_raw(task: str):
    raise NotImplementedError


def clean_text(df):
    raise NotImplementedError


def split_dataset(df, seed: int = 42):
    raise NotImplementedError


def save_processed(train, val, test, task: str):
    raise NotImplementedError


if __name__ == "__main__":
    pass
