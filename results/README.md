# Reported tables and numerical results

Start with the [figure/table index](../FIGURES_AND_TABLES.md) to locate a specific result. All files here contain aggregate results or published aggregate visuals.

| Directory | What it contains |
|---|---|
| [publication_tables](publication_tables/) | All 23 manuscript/supplement tables: `Table_1.csv`–`Table_3.csv` and `Table_S1.csv`–`Table_S20.csv` |
| [figure_source_data](figure_source_data/) | Numerical source tables for ROC summaries, subgroups, structure ablations, adjusted associations, and screening costs |
| [publication_figures](publication_figures/) | Main article figures, aggregate supplementary figures, and code-rendered diagrams (cohort flow, sites, calibration, architecture, metabolic curves, costs, and pathway) |
| [architecture_comparison](architecture_comparison/) | Backbone, stream, computational-cost, and matched DeiT comparison results |
| [figure5](figure5/) | 45 retinal-structure odds ratios and their confidence intervals |
| [vessel_pairwise_associations](vessel_pairwise_associations/) | 27 correlations per target: measured PWV and the image score |
| [frozen_representation](frozen_representation/) | 81 vessel-probe results, in long and display-ready formats |
| [shap_20mi](shap_20mi/) | Global importance by imputation and pooled across 20 imputations |
| [clinical_utility](clinical_utility/) | Screening-cost counts and net benefit at threshold 0.20 |

The publication-table CSVs preserve displayed values, multirow headings, blank merged cells, and textual confidence intervals. Their `index.json` records filenames, row counts, and checksums. They are a convenient copy of the reported tables, not participant-level datasets.

The analysis-specific CSVs contain numerical aggregate outputs at their stored precision or the article display precision specified in each table. Use these for calculations and figure rendering where indicated in the index. Figure panels based on individual predictions, SHAP values, or retinal photographs use authorized local inputs; hospital policy governs access to those data.
