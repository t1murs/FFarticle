# Packaging validation (23 November 2026)

All results below concern the code extracted from the source archive and the subset of frozen outputs supplied here.

- **Reference deterministic verification:** 15/15 PASS for V01–V15. Command: `python stage3b/reference_verification/scripts/run_verification.py`.
- **Accelerated/reference engine review:** FE01–FE06 PASS, including exact scalar RNG and domain-level comparisons. Command: `python stage3b/tests/review_fast_engine.py`. Maximum matched summary floating-point discrepancy: 4.44e-16 (declared tolerance 1e-12). A local C++ backend was built for the check; the compiled binary is intentionally not distributed.
- **Table reproduction:** manuscript Table 2 and key Table 3 counts restored exactly from `stage3b/results/primary_findings.json` using `scripts/reproduce_article_tables.py` with the Python standard library. This did not rerun simulated firm histories.
- **Archive integrity:** all manifested packaged files were verified by SHA256 and ZIP member integrity; the zip is a separate minimal distribution, not the original 214 MB package.
- **Not repeated in this packaging step:** full production simulation, complete trajectory replay, full Parquet regeneration, graphical regeneration from tables, empirical estimation, and external independent human validation. Parquet reconstruction requires a compatible installed reader (see `requirements.txt`).

**The original scientific NO-GO for the strong theory remains unchanged.** The manuscript presents only a narrower computational sensitivity analysis and retains its negative outcomes and conditional findings.
