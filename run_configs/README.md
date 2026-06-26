# Run configurations — paper index (synthetic benchmarks)

Each file is a ready-to-run config that loads an auction-type config from
`core/configs/` and fixes a single architecture. Run one with:

```bash
python main.py --setting=run_configs/<subfolder>/<config>.py
```

Architecture suffix → model (paper name):
`net` → CANet · `former` → CAFormer · `graph` → CAGraph.

Settings are `<agents>x<items>`; bundle counts are `2^items − 1`.

## `noncombinatorial/` — Table 1, Non-combinatorial (A)
Additive item values, `U[0,1]` (`ca_uniform_01`).

| Setting | CANet | CAFormer | CAGraph |
| --- | --- | --- | --- |
| 2×2 | `ca_01_net_2x2.py` | `ca_01_former_2x2.py` | `ca_01_graph_2x2.py` |
| 2×3 | `ca_01_net_2x3.py` | `ca_01_former_2x3.py` | `ca_01_graph_2x3.py` |
| 2×5 | `ca_01_net_2x5.py` | `ca_01_former_2x5.py` | `ca_01_graph_2x5.py` |

## `combinatorial/` — Table 2
Super-additive valuations (complementarity term `c_iS ~ U[-1,1]`).

### Symmetric (B) — `ca_uniform_12`
| Setting | CANet | CAFormer | CAGraph |
| --- | --- | --- | --- |
| 2×2 | `ca_net_2x2.py` | `ca_former_2x2.py` | `ca_graph_2x2.py` |
| 2×3 | `ca_net_2x3.py` | `ca_former_2x3.py` | `ca_graph_2x3.py` |
| 2×5 | `ca_net_2x5.py` | `ca_former_2x5.py` | `ca_graph_2x5.py` |

### Asymmetric (C) — `ca_uniform_12_15`
| Setting | CANet | CAFormer | CAGraph |
| --- | --- | --- | --- |
| 2×2 | `ca_12_15_net_2x2.py` | `ca_12_15_former_2x2.py` | `ca_12_15_graph_2x2.py` |
| 2×3 | `ca_12_15_net_2x3.py` | `ca_12_15_former_2x3.py` | `ca_12_15_graph_2x3.py` |
| 2×5 | `ca_12_15_net_2x5.py` | `ca_12_15_former_2x5.py` | `ca_12_15_graph_2x5.py` |

---
Run `python check_configs.py` to load every config and print a validation table.
