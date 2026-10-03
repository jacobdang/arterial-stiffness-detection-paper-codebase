.PHONY: verify test test-inference smoke
verify:
	PYTHONDONTWRITEBYTECODE=1 python scripts/paper_index.py --verify
	PYTHONDONTWRITEBYTECODE=1 python scripts/verify_public_aggregates.py
	sha256sum --quiet -c SHA256SUMS

test:
	PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider

test-inference:
	PYTHONDONTWRITEBYTECODE=1 python -c "import torch, timm, safetensors"
	PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider

smoke:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python examples/run_epoch19_cpu_smoke.py --output /tmp/pwv_epoch19_cpu_smoke_result.json
