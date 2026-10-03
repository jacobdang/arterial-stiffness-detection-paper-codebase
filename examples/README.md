# Synthetic inference example

The two 384 × 384 RGB images are deterministic generated patterns containing no patient data. `synthetic_fixture_manifest.json` records their checksums and generation settings.

```bash
PYTHONPATH=src python examples/run_epoch19_cpu_smoke.py --output /tmp/pwv_example.json
```

The example performs CPU inference with the released epoch-19 checkpoint. Its score is a software check on synthetic inputs, not a clinical performance estimate. `epoch19_cpu_smoke_result.json` contains the reference output.
