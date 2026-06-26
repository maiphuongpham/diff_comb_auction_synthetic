# from core import run_train
# import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument("--setting", type=str,
#                     help="The type of procedure corresponding to the name of config")
# parser.add_argument("--n_run", type=int, default=1,
#                     help="The number of runs of the same experiment with different random seeds")
# parser.add_argument("--n_gpu", type=int, default=0,
#                     help="The number of GPUs")
# args = parser.parse_args()


# # RAY = args.n_gpu >= args.n_run
# # if RAY:
# #     import ray
# #     from torch import cuda

# #     N_GPUS = 1 if cuda.is_available() else 0

# #     @ray.remote(num_cpus=1, num_gpus=N_GPUS)
# #     def main_remote(setting, seed):
# #         return run_train.main(setting, seed)

# #     ray.init()
# #     result_ids = []
# #     for seed in range(args.n_run):
# #         result_ids.append(main_remote.remote(args.setting, seed))
# #     ray.get(result_ids)

# # else:
# #     for seed in range(args.n_run):
#         # run_train.main(args.setting, seed)

# run_train.main(args.setting)

from core import run_train
import argparse
import importlib, os, ast, sys
from typing import Any

def _smart_cast(v: str) -> Any:
    # numbers, bools, None; otherwise keep as string
    try:
        return ast.literal_eval(v)
    except Exception:
        return v

def _set_by_dotted_key(edict_obj, dotted: str, value):
    parts = dotted.split(".")
    cur = edict_obj
    for p in parts[:-1]:
        if p not in cur or cur[p] is None:
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

parser = argparse.ArgumentParser()
parser.add_argument("--setting", type=str,
                    help="The type of procedure corresponding to the name of config")
parser.add_argument("--n_run", type=int, default=1,
                    help="The number of runs of the same experiment with different random seeds")
parser.add_argument("--n_gpu", type=int, default=0,
                    help="The number of GPUs")
# NEW: repeatable overrides like --set train.learning_rate=1e-3
parser.add_argument("--set", action="append", default=[],
                    help='Override cfg via dotted key, e.g. --set train.learning_rate=1e-3 (repeatable)')

args = parser.parse_args()

# If overrides were provided, import the config module now and mutate cfg in-place
if args.set:
    # Resolve module path from file path like "run_configs/combinatorial/ca_graph_2x2.py"
    setting_path = args.setting
    if setting_path.endswith(".py"):
        setting_path = setting_path[:-3]
    module_path = setting_path.replace("/", ".").replace("\\", ".")
    mod = importlib.import_module(module_path)  # cached in sys.modules
    if not hasattr(mod, "cfg"):
        raise RuntimeError(f"Config module '{module_path}' has no 'cfg' EasyDict.")

    cfg = getattr(mod, "cfg")
    for kv in args.set:
        if "=" not in kv:
            raise ValueError(f"--set must be key=value; got: {kv}")
        k, v = kv.split("=", 1)
        _set_by_dotted_key(cfg, k.strip(), _smart_cast(v.strip()))

# Finally, run (run_train/main will import the same module from sys.modules)
run_train.main(args.setting)
