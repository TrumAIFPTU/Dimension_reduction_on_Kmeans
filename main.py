from pathlib import Path
from src.config import get_default_configs
from src.experiments.run_experiment import run_all_sweeps
import time

def main():
    start = time.time()
    out_dir = Path("outputs")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    cfgs = get_default_configs()
    df_detail, df_summary = run_all_sweeps(cfgs, out_dir=out_dir)

    print("\n=== DONE ===")
    print("\n--- SUMMARY (mean over seeds) ---")
    print(df_summary.sort_values(["dataset", "dimred", "d"]).to_string(index=False))

    print(f"\nSaved detail: {out_dir / 'results_detail.csv'}")
    print(f"Saved summary: {out_dir / 'results_summary.csv'}")
    print(f"Figures: {out_dir / 'figures'}")
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
