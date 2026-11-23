# External dataset provenance (contextual inputs only)

This repository supports the computational article **Inherited Advice in Family Firms: A Computational Study of Information Access, Verification and Successor Learning**.

## Important distinction

The two external family-firm datasets below were inspected and audited during research feasibility work (Study 1 / Stage 1). They were **not used to estimate or calibrate** the behavioral parameters, to generate the synthetic decision histories, or to compute the simulation results presented in the article. The computational experiments use a separately specified, scenario-assumed model. The data are cited here to document the scope and provenance of the earlier source/construct feasibility audit. No country-level empirical inference from these two datasets is reported in the simulation article.

## Dataset citations and version-specific DOIs

**China (Mendeley Data; 2012–2022).**

Song, S. (2026). *Digital Transformation and Family Firm Internationalization: Evidence from China* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/xjcbfhxs69.1

- Repository landing page: https://data.mendeley.com/datasets/xjcbfhxs69/1
- Files examined in the original source audit: `Replication.dta`, `Replication.do`, `README.docx`.
- Substantive panel: 9,666 firm-year observations representing 1,116 Chinese listed family firms, 2012–2022.
- License shown by repository: Creative Commons Attribution 4.0.
- Related research article (different DOI): Song et al. (2026), *Digital transformation and family firm internationalization: Evidence from China*, *Economic Modelling*. https://doi.org/10.1016/j.econmod.2026.107781

**Spain (Zenodo; 2016 survey, financial records for 2015).**

Sánchez Marín, G., Baixauli-Soler, J. S., Lozano-Reina, G., Belda-Ruiz, M., & Peláez-León, J. D. (2025). *Family firms: SEW, HRM practices, and financial indicators* (Version 2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17723051

- Repository landing page: https://zenodo.org/records/17723051
- Files examined in the original source audit: `Dataset and codebook_FF HRM FIN (2016) vfinal.xlsx`, `Questionnarie for executives_HRM in FF (english).pdf`.
- 508 Spanish medium-sized family firms. Survey collected in 2016; source financial records relate to 2015.
- License shown by repository: Creative Commons Attribution 4.0.
- Parent/concept DOI for the dataset version family: https://doi.org/10.5281/zenodo.17225495
- Related data paper (different DOI): Baixauli-Soler, J. S., Belda-Ruiz, M., Lozano-Reina, G., Peláez-León, J. D., & Sánchez-Marín, G. (2026). *Dataset on Spanish medium-sized family firms: Linking socioemotional wealth, HRM practices, and financial indicators*. *Data in Brief, 65*, 112488. https://doi.org/10.1016/j.dib.2026.112488

## Redistribution

The original files are available through their own repositories. This project does **not** mirror the external microdata. Follow each source's current licensing and provenance requirements if you plan to redistribute copies. The original source audits include caveats about variable meaning, missingness, and suitability for computational calibration. They should not be interpreted as reproducing the source studies' econometric findings.

## Reproducing the simulation article

Use the model, code, frozen configurations, parameter-design coordinates, selected analysis-ready Parquet tables, figure data, dependency versions, verification tests, and run manifests provided separately with this article. The external China and Spain files are **not required** to reproduce the simulation findings.

Provenance checked: 23 November 2026.
