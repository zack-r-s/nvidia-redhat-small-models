# Mark III NeMo Gym task

`scripts/prepare_gym_mark_iii.py` freezes the two task files in `data/` from the seed-disjoint Mark I PyAT correction cases. Run it without arguments to verify the exact committed bytes. The server uses one local PyAT fixture per rollout; it does not call Osprey or the demo fixture process.

Install NeMo Gym 0.4.0 in `var/gym-cli`, then run `gym env test --resources-server hackring_mark_iii` from `gym/`. With a pinned model listening on a local vLLM endpoint, set `VLLM_BASE_URL`, `VLLM_MODEL_NAME`, `VLLM_MODEL_SOURCE`, and `VLLM_MODEL_REVISION`, then run `scripts/run_gym_mark_iii.sh direct --num-repeats 4` or `scripts/run_gym_mark_iii.sh skill --num-repeats 4` from the repo root. Results and full model/tool metadata go under `var/gym/mark-iii/<track>/<model>/<run>/`.
