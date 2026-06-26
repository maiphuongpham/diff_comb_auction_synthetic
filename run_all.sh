#!/usr/bin/env bash
# Run the synthetic-benchmark experiments (Section 5, Tables 1 & 2).
# Configs are grouped by paper table (see run_configs/README.md).
#
# Usage:
#   ./run_all.sh                   # run every experiment sequentially
#   ./run_all.sh noncombinatorial  # run only one group (noncombinatorial|combinatorial)
#   ./run_all.sh -n                # dry run: print commands without executing
set -euo pipefail

cd "$(dirname "$0")"

# --- environment ---
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"   # silence matplotlib cache warning
PY="${PY:-python}"

# --- args ---
DRY=0
GROUP="all"
for a in "$@"; do
  case "$a" in
    -n|--dry-run) DRY=1 ;;
    noncombinatorial|combinatorial) GROUP="$a" ;;
    *) echo "Unknown arg: $a"; exit 1 ;;
  esac
done

run() {  # run <group> <config-path>
  [[ "$GROUP" == "all" || "$GROUP" == "$1" ]] || return 0
  echo ">>> [$1] $2"
  (( DRY )) && return 0
  "$PY" main.py --setting="$2"
}

# ===== Table 1 — Non-combinatorial (A), U[0,1] =====
run noncombinatorial run_configs/noncombinatorial/ca_01_net_2x2.py
run noncombinatorial run_configs/noncombinatorial/ca_01_net_2x3.py
run noncombinatorial run_configs/noncombinatorial/ca_01_net_2x5.py
run noncombinatorial run_configs/noncombinatorial/ca_01_former_2x2.py
run noncombinatorial run_configs/noncombinatorial/ca_01_former_2x3.py
run noncombinatorial run_configs/noncombinatorial/ca_01_former_2x5.py
run noncombinatorial run_configs/noncombinatorial/ca_01_graph_2x2.py
run noncombinatorial run_configs/noncombinatorial/ca_01_graph_2x3.py
run noncombinatorial run_configs/noncombinatorial/ca_01_graph_2x5.py

# ===== Table 2 — Combinatorial, Symmetric (B), ca_uniform_12 =====
run combinatorial run_configs/combinatorial/ca_net_2x2.py
run combinatorial run_configs/combinatorial/ca_net_2x3.py
run combinatorial run_configs/combinatorial/ca_net_2x5.py
run combinatorial run_configs/combinatorial/ca_former_2x2.py
run combinatorial run_configs/combinatorial/ca_former_2x3.py
run combinatorial run_configs/combinatorial/ca_former_2x5.py
run combinatorial run_configs/combinatorial/ca_graph_2x2.py
run combinatorial run_configs/combinatorial/ca_graph_2x3.py
run combinatorial run_configs/combinatorial/ca_graph_2x5.py

# ===== Table 2 — Combinatorial, Asymmetric (C), ca_uniform_12_15 =====
run combinatorial run_configs/combinatorial/ca_12_15_net_2x2.py
run combinatorial run_configs/combinatorial/ca_12_15_net_2x3.py
run combinatorial run_configs/combinatorial/ca_12_15_net_2x5.py
run combinatorial run_configs/combinatorial/ca_12_15_former_2x2.py
run combinatorial run_configs/combinatorial/ca_12_15_former_2x3.py
run combinatorial run_configs/combinatorial/ca_12_15_former_2x5.py
run combinatorial run_configs/combinatorial/ca_12_15_graph_2x2.py
run combinatorial run_configs/combinatorial/ca_12_15_graph_2x3.py
run combinatorial run_configs/combinatorial/ca_12_15_graph_2x5.py

echo "Done."
