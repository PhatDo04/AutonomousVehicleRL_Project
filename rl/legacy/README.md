# Legacy single-agent

Chỉ dùng với `models/single_agent/Main_Traffic_SingleAgent.gaml` và experiment `TrafficSingleHeadless`.

Pipeline đồ án chính: `rl/run_experiments.py` (MARL) + `models/Main_Traffic.gaml`.

```powershell
gama-headless.bat -socket 1001
python rl/legacy/train_single.py --algo dqn --timesteps 20000 --port 1001
python rl/legacy/evaluate_single.py --algo dqn --model outputs/models/dqn_seed0_20k.zip
```

Baseline Greedy: `python rl/baselines.py` (cùng archive GAML).
