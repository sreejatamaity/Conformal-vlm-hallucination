# Post-Hoc Conformal Prediction for Hallucination Detection in Vision-Language Models

Split-Conformal Prediction wrapped around a frozen LLaVA-1.5-7B model, giving a **distribution-free, statistically-guaranteed** signal for when the model's answer should not be trusted — evaluated on the POPE-Adversarial hallucination benchmark.

## Overview

Vision-Language Models produce confident but sometimes factually wrong answers about image content. Raw softmax probability is not a reliable confidence signal — there's no guarantee a model's stated 90% confidence corresponds to a 90% chance of being correct.

This project applies **split-conformal prediction** as a post-hoc wrapper: given any frozen model and a small calibration set, it builds prediction sets — `{Yes}`, `{No}`, or `{Yes, No}` — with a provable guarantee that the true label is included at least $1-\alpha$ of the time, regardless of how well-calibrated the underlying model actually is.

## Key Results

Evaluated on **POPE-Adversarial** (COCO val2014), the hardest of POPE's three splits, with LLaVA-1.5-7B in 4-bit quantization:

| Metric | Value |
|---|---|
| Calibration / test size | 1,000 / 2,000 |
| Target coverage (1 − α) | 90% |
| **Empirical coverage** | **91.45%** |
| Average prediction set size | 1.195 |
| Confident singleton predictions | 80.5% |
| Flagged as uncertain (potential hallucination) | 19.5% |
| Raw point-prediction accuracy (no CP) | 83.35% |

All numbers are generated directly by the notebook and saved to [`results/results_summary.json`](results/results_summary.json) — nothing here is hand-transcribed.

## Repository Structure

```
├── conformal_prediction.ipynb   # complete, self-contained pipeline (Colab/Kaggle-ready, T4 GPU)
├── figures/                     # all output plots from the run reported below
│   ├── calibration_score_distribution.png
│   ├── coverage_vs_alpha.png
│   ├── prediction_set_size_distribution.png
│   ├── qualitative_examples_grid.png
│   └── sample_image.png
└── results/                     # raw output artifacts from the run
    ├── results_summary.json     # final aggregated metrics and configuration
    ├── calib_checkpoint.json    # all 1,000 raw calibration non-conformity scores
    └── test_checkpoint.json     # all 2,000 raw per-example test results
```

## Reproducing the Results

1. Open `conformal_prediction.ipynb` in Google Colab or Kaggle.
2. Set the runtime to a **T4 GPU**.
3. Run all cells top to bottom. Calibration and test loops checkpoint progress every 50 samples, so an interrupted run can be resumed by simply re-running the same cell.
4. The final cell saves `results_summary.json` with every metric reported above.

Random seeds (Python, NumPy, PyTorch/CUDA) are fixed at 42.

## Method Summary

1. **Uncertainty extraction:** a single forward pass through LLaVA extracts next-token logits at the answer position; a 2-class softmax over only the `Yes`/`No` tokens gives a binary probability — no autoregressive decoding required.
2. **Calibration:** the non-conformity score $s(x,y) = 1 - \hat p(y \mid x)$ is computed on 1,000 held-out calibration examples, and the finite-sample-corrected quantile $\hat q$ is computed at the target miscoverage rate $\alpha$.
3. **Prediction sets:** for each test example, both labels whose probability exceeds $1-\hat q$ are retained. A two-label set flags the example as uncertain.

## Limitations

This is a single calibration/test split (not a multi-fold evaluation), one model, and one POPE split (adversarial). A theoretical check (Beta-distribution concentration of split-conformal coverage) confirms the observed 91.45% is a statistically unremarkable draw from the expected sampling distribution rather than an anomalous result — full discussion in the paper (see Status below).

## Acknowledgements

Built on [LLaVA-1.5](https://github.com/haotian-liu/LLaVA), evaluated on [POPE](https://github.com/AoiDragon/POPE), using conformal prediction theory from Vovk et al. (2005) and Angelopoulos & Bates (2023).

## Status

Preprint available on Zenodo: [10.5281/zenodo.23180933](https://doi.org/10.5281/zenodo.23180933). arXiv submission in progress.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@misc{maity2026conformalvlm,
  author    = {Maity, Sreejata},
  title     = {Post-Hoc Conformal Prediction for Hallucination Detection in Vision-Language Models: A Distribution-Free Coverage Guarantee on {POPE-Adversarial}},
  year      = {2026},
  month     = oct,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23180933},
  url       = {https://doi.org/10.5281/zenodo.23180933},
  note      = {Preprint}
}
```
