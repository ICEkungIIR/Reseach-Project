"""
tokenizer_analysis.py

Purpose:
    Tokenize the same processed dataset with both the WangchanBERTa
    tokenizer and the PhayaThaiBERT tokenizer, and compute fertility
    (tokens per word / tokens per character) per example and per token.

    Output is stored alongside the processed data so RQ3's token-level
    error analysis can join on it later without re-tokenizing.

Planned functions:
    - compute_fertility(text: str, tokenizer) -> float
    - analyze_dataset(dataset, tokenizer_name: str) -> pandas.DataFrame
    - compare_tokenizers(dataset) -> pandas.DataFrame  # side-by-side stats

Not yet implemented — scaffold only.
"""

def compute_fertility(text: str, tokenizer):
    raise NotImplementedError


def analyze_dataset(dataset, tokenizer_name: str):
    raise NotImplementedError


def compare_tokenizers(dataset):
    raise NotImplementedError


if __name__ == "__main__":
    pass
