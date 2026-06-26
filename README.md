# Differentiable Combinatorial Auction Design — Synthetic Benchmarks

This repository contains the code to reproduce the **synthetic computational
results (Section 5, Tables 1 & 2)** of *Differentiable Combinatorial Auction
Design: Optimizing Revenue–Welfare–Fairness Frontiers with Graph Attention
Networks*.

It implements three differentiable combinatorial-auction architectures —
**CANet**, **CAFormer**, and the graph-based **CAGraph** — and trains them with a
regret-budget procedure to learn approximately strategy-proof, feasible-by-
construction mechanisms. The airport slot-allocation case study (Section 6) and
appendix case studies are **not** included here; they live in the private full
repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The VCG baseline is solved with **Gurobi** (`gurobipy`), which requires a valid
license (a free academic license is available from gurobi.com). All other
components run on CPU or GPU with PyTorch.

## Architectures

| Suffix in config name | Model | Paper name |
| --- | --- | --- |
| `net` | `CANet` | CANet |
| `former` | `CANetFormer` | CAFormer |
| `graph` | `CAGCN` | CAGraph |

## Experiments

Configs live in `run_configs/`, grouped by paper table:

| Folder | Paper location | Distribution |
| --- | --- | --- |
| `run_configs/noncombinatorial/` | Table 1 — Non-combinatorial **(A)** | `ca_uniform_01` (additive `U[0,1]`) |
| `run_configs/combinatorial/` | Table 2 — Symmetric **(B)** & Asymmetric **(C)** | `ca_uniform_12` / `ca_uniform_12_15` |

See [run_configs/README.md](run_configs/README.md) for the per-config index and
[core/configs/README.MD](core/configs/README.MD) for every tunable parameter.

### Run a single experiment

```bash
python main.py --setting=run_configs/combinatorial/ca_graph_2x2.py
```

Override any config value without editing files (repeatable):

```bash
python main.py --setting=run_configs/noncombinatorial/ca_01_net_2x2.py \
  --set train.learning_rate=1e-3 --set train.max_iter=100000
```

### Run a group / everything

```bash
./run_all.sh                  # all synthetic experiments sequentially
./run_all.sh noncombinatorial # only Table 1
./run_all.sh -n               # dry run: print commands without executing
```

### Validate configs

```bash
python check_configs.py       # loads every config and prints agents/items/bundles/distribution
```

## Logging

Training statistics are printed to stdout and logged to TensorBoard (and
Weights & Biases if enabled). View with:

```bash
tensorboard --logdir runs
```

## License

See [LICENSE](LICENSE).
