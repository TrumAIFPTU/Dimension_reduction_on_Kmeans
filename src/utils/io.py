import pandas as pd
from pathlib import Path

def save_results_csv(rows, out_path: Path):
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df
