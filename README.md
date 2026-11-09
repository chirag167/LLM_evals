# A Pilot Study -- Statistical and systematic uncertainty in an LLM benchmark comparison

I took two open-weight language models, scored them on the same 1,500 MMLU questions under 24 different but equally reasonable evaluation setups, and built an uncertainty budget for the difference between them.

The question: when a benchmark says model A beats model B by some number of points, how much of that number depends on choices the evaluator made, and how does that compare with ordinary sampling noise?

## Background

Two earlier papers each cover half of this.

Miller (2024) treats benchmark questions as a random draw from a larger population and gives standard errors for a single score and for the paired difference between two models, including a correction for questions that come in related groups (clustered standard errors). It does not consider how evaluation choices change the result.

Sclar et al. (2024) show that changing prompt formatting without changing meaning (separators, spacing, capitalization, option labels) can move accuracy by many points, and that the ranking of two models can reverse depending on the format. They test individual comparisons with paired McNemar tests. They report the spread across formats as a range and a standard deviation, but they do not separate that spread into the part caused by sampling noise and the part caused by the format itself.

In experimental physics those two parts are always reported separately, as statistical and systematic uncertainty. This project applies that bookkeeping to a benchmark comparison.

## Setup

Models: `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`, run locally with Hugging Face `transformers`, greedy decoding.

Questions: 1,500 questions drawn at random (seed 0) from the MMLU test split (Hendrycks et al., 2021). Both models answer exactly the same questions, so every comparison is paired. The 5-shot examples come from MMLU's dev split, the same five per subject every time.

Four evaluation choices ("knobs"), fixed in `config.json` before any real model was run:

| Knob | Settings |
|---|---|
| Prompt template | `plain` (Question / options / Answer:) or `instructed` (adds a one-line instruction to reply with the letter only) |
| Few-shot examples | 0 or 5 |
| Option order | as published, or shuffled with a fixed permutation per question |
| Answer extraction | first character of the reply, first standalone A-D found by regex, or the highest-probability letter token after `Answer:` |

That gives 2 × 2 × 2 × 3 = 24 configurations. Extraction is applied to the same model reply, so each model is called 8 × 1,500 = 12,000 times.

## Method

For each configuration, the per-question difference d = (0.5B correct) − (1.5B correct) is computed, and its mean Δ is the measured gap. Each Δ gets a paired standard error (Miller, Eq. 7) and a paired standard error clustered by MMLU subject (Miller, Eq. 8).

The statistical uncertainty quoted below is the median clustered standard error across configurations. The systematic uncertainty is the standard deviation of Δ across the 24 configurations. A bootstrap over questions (2,000 resamples, the same resample applied to every configuration so that pairing is kept) gives the uncertainty on that systematic estimate, and a combined distribution where each draw takes a random set of questions and a random configuration.

## Results

| Quantity | Value (accuracy points) |
|---|---|
| Δ = 0.5B − 1.5B, mean over configurations | −16.2 |
| Range of Δ across configurations | −22.3 to −12.7 |
| Statistical uncertainty (median clustered SE) | 2.2 |
| Systematic uncertainty (spread across configurations) | 2.8, bootstrap 68% interval 2.7 to 3.4 |
| Combined 95% interval (bootstrap) | −22.5 to −11.3 |
| Configurations where 1.5B wins / is significant | 24 of 24 / 24 of 24 |

Shift in Δ attributable to each knob (half the difference between its highest and lowest setting, averaged over the other knobs):

| Knob | Shift (points) |
|---|---|
| Few-shot examples | 1.6 |
| Answer extraction | 1.4 |
| Option order | 1.1 |
| Prompt template | 0.7 |

Full per-configuration output is in `analysis_output.txt`.

## What this shows

The larger model wins under every setup, as expected for a threefold size difference. This run serves as a control: the method gives a clean answer when the answer is clean.

The systematic uncertainty (2.8 points) is about as large as the statistical uncertainty (2.2 points). Reporting only the statistical error bar would understate the real uncertainty on the gap, and the size of the gap itself ranges from 13 to 22 points depending on reasonable choices. For two models a few points apart, an effect of this size could decide which one appears better.

Clustering matters on real data. The clustered standard error is up to about twice the naive paired one, because questions within an MMLU subject are correlated, as Miller predicts.

Text-based extraction widens the gap relative to probability-based extraction (−17.3 for first-character, −14.5 for logprob). One possible reason is that the smaller model more often replies with something other than a bare letter when given no examples. I have not checked this yet.

## Limitations and next steps

- One model pair, one benchmark, 1,500 questions.
- The per-knob shifts of about one point are smaller than the statistical error of a single configuration. I have not yet tested whether each is distinguishable from noise.
- The 24 configurations share questions and use greedy decoding, so most noise cancels between them, but not necessarily all of it. A split-half test (does the pattern across configurations repeat on two random halves of the questions?) would show how much of the 2.8 points is truly systematic. This is the step that neither paper above takes, and it is next.
- Count extraction failures per model and setting, to test the explanation for the extraction effect.
- Repeat on a pair of models whose accuracies differ by only a few points, where the systematic term could change the ranking.
- Compare each model's absolute accuracy with published MMLU numbers as a sanity check.

## Running it

```
conda create -n llm_evals python=3.11
conda activate llm_evals
python -m pip install -r requirements.txt

python run_eval.py config.json            # writes results.csv
python analyze.py results.csv "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct"
```

Models and the MMLU data download automatically from Hugging Face on first use. To test the pipeline without downloading anything, set `"models": ["fake:model_A", "fake:model_B"]` and `"dataset": "synthetic"` in `config.json`; these are simulated models with built-in quirks, not real ones.

On Apple-silicon Macs, make sure Python runs natively (`python -c "import platform; print(platform.machine())"` should print `arm64`); an Intel build of Anaconda installs an outdated PyTorch.

## Files

- `config.json`: models, dataset, seed and the knob settings
- `run_eval.py`: builds prompts, queries models, grades answers, writes `results.csv` (one row per model, configuration and question)
- `analyze.py`: paired and clustered errors, systematic spread, per-knob budget, bootstrap
- `results.csv`, `analysis_output.txt`: the outputs reported above

## References

- E. Miller, "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations," arXiv:2411.00640 (2024).
- M. Sclar, Y. Choi, Y. Tsvetkov, A. Suhr, "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting," ICLR 2024, arXiv:2310.11324.
- D. Hendrycks et al., "Measuring Massive Multitask Language Understanding," ICLR 2021, arXiv:2009.03300.
