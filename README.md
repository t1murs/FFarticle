# Inherited Advice in Family Firms — reproducibility materials

Reproducibility companion for the manuscript *Inherited Advice in Family Firms: A Computational Study of Information Access, Verification and Successor Learning* (Econviews working draft v0.2).

**Scope:** Model-based, scenario-assumed results. No real generative AI is implemented, no real successor behavior is measured, and no ownership-transfer or economic-resilience effects are estimated. This is a post-results publication reframing; original negative findings and original parameters were retained.

## Quick start

The following requires Python 3.12+ for the pinned dependencies. The small, standard-library table reproduction can run without installing the scientific stack:

```bash
python scripts/reproduce_article_tables.py
python scripts/verify_archive_hashes.py
```

The first command writes `output/Table_2.csv`, `output/Table_3.csv` and prints the original manuscript counts. They are reconstructed from **saved audited summaries**, not from raw simulated histories.

For original deterministic model checks and Parquet inspection, create a clean Python environment and install pinned requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python stage3b/reference_verification/scripts/run_verification.py
```

The reference test script executes the 15 original V01–V15 structural tests and writes local test logs. No new production simulation is run.

On Linux with g++ supporting C++17 and OpenMP you may also run the original fast-vs-reference equivalence test:

```bash
python stage3b/tests/review_fast_engine.py
```

This optional test builds `stage3b/src/family_ai_fast/_backend.so` locally. Compiled binaries are excluded from this archive. The source includes NumPy-derived PCG64 code; preserve its license notice.

## What is in this repository?

- `stage3b/reference_verification/`: exact baseline NumPy model and structural verification suite.
- `stage3b/src/`: accelerated implementation and storage/analysis helpers (source only).
- `stage3b/config/`: frozen numerical specification, parameter design, Sobol coordinates, and precision/amendment records.
- `stage3b/results/`: verified human-readable findings and selected machine-readable Parquet results.
- `figures/`: exact two PNG panels embedded in manuscript v0.2.
- `stage3b/figures/data/`: source data from related original Stage 3B figures (their layout is not identical to the manuscript graphics).
- `DATA_SOURCES.md`: DOI citations and role of original Chinese and Spanish source datasets; the external data are not distributed here.
- `REPRODUCIBILITY_SCOPE.md`: precise limits of what this minimal repository can reproduce.
- `VALIDATION_REPORT.md`: actual tests executed on this extracted subset.
- `SOURCE_FILE_HASHES.csv`: SHA256 hashes of files selected from the original archived bundle.
- `SHA256SUMS.txt`: SHA256 hashes of all files included (except the list itself).

## Published numerical interpretation (SIMULATED)

- Pure improved access, 0.02 joint early gain/later loss: **0 / 354** eligible points for each of two learning rules.
- At a zero threshold, direction-only divergence: **163 / 354** and **162 / 354**.
- D–C checking changed post-shock correctness by > +0.02 in **83 / 768** and **79 / 768**, or < −0.02 in **48 / 768** and **64 / 768**, under the two learning rules.
- Information-equivalent archive/AI proxy modes are identical by construction; this does not imply equivalence of actual AI systems.
- The original registered strong family-specific contribution was judged scientifically insufficient; the present paper makes a narrower, retrospective claim.

## Reproducing figures

The `figures/` directory contains the exact two manuscript images, extracted directly from the v0.2 DOCX. Selected point-level Parquet files and the original Stage 3B figure-data files are retained for independent replotting. Figure-generating scripts in the original complete research bundle require additional full aggregated data and are not advertised as runnable from this minimal distribution.

## Data and code availability

For both publicly hosted contextual datasets and their versioned DOI, see `DATA_SOURCES.md`. Neither dataset was used to calibrate the simulation or generate virtual firms. Raw simulation histories (~50 GB) and other large audits remain in the original Stage 3B research archive. This GitHub package is not sufficient to reconstruct every full trajectory from stored outputs.

**Anonymized review:** A publicly accessible repository linked from a double-anonymized journal submission may reveal ownership, account identity and timestamps. Follow the journal's anonymous-supplement policy and consider an anonymized repository snapshot until peer review is finished.

**License:** Original project code rights and a distribution license must be confirmed by the human author before public release; see `LICENSE_README.md`. NumPy-derived material retains its source notice.
