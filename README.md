# Effect of LoRA Rank on Fine-tuning Performance for Thai Language Under Tokenization Constraints

**Authors:** ศรัณยู เจริญผล, สหรัฐ งามเลิศ

## Research Gap
Prior work shows LoRA and full fine-tuning learn structurally different
solutions (LoRA introduces "intruder dimensions") but this is only tested
in English. Separately, Thai NLP work shows high-fertility tokenizers
(no word boundaries) hurt downstream performance. No work connects these
two threads.

## Research Questions
- **RQ1:** How does the LoRA vs. full fine-tuning F1-score gap on Thai
  text classification / NER compare to the gap reported in English?
- **RQ2:** How does LoRA rank (r = 4, 8, 16, 32) relate to performance
  gap size across tokenizers of different fertility (WangchanBERTa vs.
  PhayaThaiBERT)?
- **RQ3:** Are high-fertility tokens affected more by LoRA than typical
  tokens, measured via token-level error?

## Hypotheses
- **H0:** The Thai LoRA-vs-full-FT performance gap is not significantly
  different from the English gap, regardless of tokenizer.
- **H1:** The Thai gap is significantly larger than the English gap,
  especially with high-fertility tokenizers.

## Repo Structure
```
data/            raw + processed datasets, data card
src/             data prep, tokenizer analysis, training, eval scripts
configs/         YAML configs per model/rank/task combination
experiments/     run outputs (logs, checkpoints, metrics) — gitignored
notebooks/       EDA and error analysis
results/         final tables and figures for the report
slurm/           HPC job submission templates
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Status
- [x] Research question, PICOC, FINER, SMART, synthesis matrix finalized
- [ ] Repo scaffold (this commit)
- [ ] Data prep + tokenizer fertility analysis
- [ ] LoRA + full FT training runs
- [ ] Token-level error analysis (RQ3)
- [ ] Significance testing + practical guideline write-up
