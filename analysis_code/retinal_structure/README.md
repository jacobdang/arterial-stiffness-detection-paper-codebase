# Retinal-structure score associations

Figure 5 relates 15 retinal-structure prediction scores to arterial stiffness in 7,331 internal-test participants. The 45 odds ratios and confidence intervals are in [figure5_associations.csv](../../results/figure5/figure5_associations.csv). They can be plotted without participant data or model fitting:

```bash
python analysis_code/figures/plot_figure5.py --input results/figure5/figure5_associations.csv --output-dir OUTPUTS/figure5
```

## Clinical inputs

Copy `config.example.yaml` into the authorized environment and update the paths and source-column mappings. The score table has one row per participant, `participant_id`, binary `as_event` (baPWV ≥1,400 cm/s), and the 15 named probability scores. The clinical table uses the same participant identifiers, age in years, sex, diabetes duration, systolic blood pressure, BMI, smoking, drinking, coronary heart disease, HbA1c, and UACR. Keep UACR on its measurement scale; the analysis does not apply a logarithmic transformation. Coronary heart disease is a separate clinical indicator.

The completed clinical table supplies the existing completed age, systolic blood pressure, BMI, and diabetes-duration values where the original measurement is absent, and the sex, smoking, and drinking assignments. The example recovery constants convert its normalized continuous fields to their measurement scales. If completed values are already on those scales, set each offset to 0 and divisor to 1. Preserve the study's clinical category assignments. Remaining missing continuous covariates use their observed cohort means; missing coronary heart disease indicators are assigned 0. Every input must describe the same participant cohort. The command checks identifiers, outcomes, and the configured participant/event counts.

To use an already assembled clinical input, set `completed_clinical: null` and supply the completed clinical assignments in `clinical`. Participant-level inputs and intermediate files must remain within the approved institutional environment.

## Analysis commands

The R dependencies are `yaml`, `readxl` (for Excel inputs), `broom`, and `pROC` (for Table S10).

```bash
# Prepare and inspect the inputs without fitting statistical models.
Rscript analysis_code/retinal_structure/run_analysis.R AUTHORIZED_DATA/retinal_structure/config.yaml --prepare-only

# Estimate the Figure 5 associations.
Rscript analysis_code/retinal_structure/run_analysis.R AUTHORIZED_DATA/retinal_structure/config.yaml

# Also calculate the Table S10 discrimination comparisons.
Rscript analysis_code/retinal_structure/run_analysis.R AUTHORIZED_DATA/retinal_structure/config.yaml --discrimination
```

The analysis commands fit logistic regression models when executed. Each retinal score is centered and scaled using its sample standard deviation in the full analysis cohort and fitted separately. HbA1c and UACR are also centered and scaled. Model 0 includes the score alone; Model 1 adds age and sex; Model 2 adds diabetes duration, systolic blood pressure, BMI, smoking, drinking, coronary heart disease, HbA1c, and UACR. Confidence intervals use the likelihood-profile method; coefficient P values use Wald tests. Benjamini–Hochberg adjustment covers the 15 score tests within each model.

`--discrimination` compares each Model 2 with its clinical-covariate-only model using paired DeLong and likelihood-ratio tests. It writes `table_s10_discrimination.csv`. This is an exploratory analysis in the same internal-test cohort. The displayed article results are supplied in [Table_S10.csv](../../results/publication_tables/Table_S10.csv).

Figure S5 concerns metabolic outcomes. Its submitted aggregate curves are supplied in [publication_figures](../../results/publication_figures/); see its entry in the [figure/table guide](../../FIGURES_AND_TABLES.md) for the clinical outcome and figure sources.
