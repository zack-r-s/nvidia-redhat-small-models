# Training data and model releases

Dataset and published model tags use the same version. The local `training/vNNN/` directories retain the chronological training-run numbers; they are one behind the dataset/model tags because the first raw Xopt collection is `datasets/v001/` and has no trained checkpoint.

| Data version | Training record | Published model | Held-out local PyAT evaluation |
| --- | --- | --- | --- |
| [`v001`](../datasets/v001/manifest.json): raw Xopt traces | — | — | — |
| [`v002`](https://huggingface.co/datasets/zack-r-s/hackring-pyat-control/tree/v002): Osprey/PyAT control conversations, including the GPT-augmented SFT export | [`training/v001`](v001/README.md) · [results](v001/results.json) | [`model v002`](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M/tree/v002) | [8-case report](../evals/pyat-v001/liquid-v001/report.json) · [56-case report](../evals/pyat-v002/liquid-v001/report.json) |
| [`v003`](https://huggingface.co/datasets/zack-r-s/hackring-pyat-control/tree/v003): canonical served examples plus measured local PyAT examples | [`training/v002`](v002/README.md) · [results](v002/results.json) | [`model v003`](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M/tree/v003) | [56-case report](../evals/pyat-v002/liquid-v002/report.json) · [paired comparison](../evals/pyat-v002/comparison.json) |

The two model checkpoints both start from `LiquidAI/LFM2.5-230M`; model v003 is not initialized from model v002. The model tags name their **input dataset version**, not the chronological local training directory. The original model `v001` tag remains as a legacy alias for the dataset-v002 checkpoint, and `legacy-training-v002` preserves the former model-`v002` reference to the dataset-v003 checkpoint. Use the matching `v002` and `v003` tags for new work.

The [dataset repository](https://huggingface.co/datasets/zack-r-s/hackring-pyat-control) and [model repository](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M) are public. Generated weights stay outside Git; the tracked results files contain their SHA-256 hashes and training configuration. The 56-case local PyAT evaluation passed 21/56 for model v002 and 20/56 for model v003, so the newer checkpoint is not established as a better controller.
