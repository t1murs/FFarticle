# Reproducibility boundary

This is a **minimal, selected-data public repository**, derived without modification from the audited Stage 3B computational package and the v0.2 Econviews manuscript figures. The source model code and frozen parameter values are preserved; the two manuscript figures are extracted without modification.

- Included: scalar reference model, compiled backend source (no portable binary claimed), deterministic verification suite, frozen model specification, joint Sobol coordinates, selected result tables, original research findings in JSON, two manuscript figures and hashes.
- Not included: complete 297 MB uncompressed experiment manifest, raw firm-round trajectories (~50 GB), huge aggregate contrast/condition matrices, trial output logs, manuscript title page, or the external Chinese/Spanish microdata.
- `scripts/reproduce_article_tables.py` **recreates manuscript Table 2 and Table 3 counts from saved audited Stage 3B findings**. This is a result-level reproduction; it does not rerun the original stochastic histories. Raw trajectory-level recomputation requires the full archived Stage 3B package.
- `stage3b/reference_verification/scripts/run_verification.py` runs the original 15 deterministic tests (fresh test outputs are generated in the local working tree).
- The included fast backend is an optional, source-level path for computing further runs. It compiles on Linux with `g++`/OpenMP. It has not been shipped as a portable executable.
- Selected Parquet files preserve sample-level point estimates, classifications, and figure-data provenance. They require `pyarrow` (or another compatible Parquet engine) for reading. They are not needed to run the standard-library manuscript-table script.
- To rebuild **all** primary aggregates, confidence calculations or trajectories exactly, obtain and verify the original `Stage3B_Complete_Reproducibility_Package.zip` rather than claiming this compact collection suffices.

Parameter cells are scenarios, not observations of actual family firms. No generative AI model was trained or evaluated. The empirical China/Spain datasets were audited for feasibility only and did not calibrate any parameter or generate synthetic histories. Study 1's stronger scientific claim previously received a NO-GO; this is a retrospective narrower computational article.
