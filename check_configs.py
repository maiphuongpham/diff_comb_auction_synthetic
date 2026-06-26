import os
import glob
import sys

# Ensure core and run_configs are in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.utils import get_objects

def main():
    config_files = sorted(glob.glob("run_configs/**/*.py", recursive=True))
    print(f"Found {len(config_files)} configuration files.")
    print("-" * 140)
    print(f"{'Config File':<50} | {'Setting':<25} | {'Arch':<15} | {'Agents':<6} | {'Items':<6} | {'Bundles':<8} | {'Dist':<20}")
    print("-" * 140)
    
    errors = 0
    for path in config_files:
        try:
            cfg, _, _, _ = get_objects(path)
            
            # Extract attributes
            setting = cfg.setting if hasattr(cfg, 'setting') else 'N/A'
            arch = cfg.architecture if hasattr(cfg, 'architecture') else 'N/A'
            agents = cfg.num_agents if hasattr(cfg, 'num_agents') else 'N/A'
            items = cfg.num_items if hasattr(cfg, 'num_items') else 'N/A'
            bundles = cfg.num_bundles if hasattr(cfg, 'num_bundles') else 'N/A'
            dist = cfg.distribution_type if hasattr(cfg, 'distribution_type') else 'N/A'
            
            print(f"{path:<50} | {setting:<25} | {arch:<15} | {agents:<6} | {items:<6} | {bundles:<8} | {dist:<20}")
        except Exception as e:
            print(f"ERROR loading {path}: {e}")
            errors += 1

    print("-" * 140)
    print(f"Validation completed. Total errors: {errors}")

if __name__ == "__main__":
    main()
