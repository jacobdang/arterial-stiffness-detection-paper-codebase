# Frozen retinal-image representation

This analysis evaluates whether a fixed image representation contains information associated with 27 measured retinal-vessel features. It compares the 2,048-dimensional representation, scalar image score, and clinical predictors using nested cross-validation. The image model remains fixed.

Edit `config.example.json` with authorized local input paths and SHA-256 checksums. Input prediction files must be explicitly trusted. Run `python analysis_code/frozen_representation/run_analysis.py --help` for output and configuration options. `PROTOCOL.md` describes the analysis definitions.

Aggregate probe results supporting Table S13 are provided in `results/frozen_representation/`. Running the analysis fits statistical probe models and requires the corresponding participant-level inputs.
