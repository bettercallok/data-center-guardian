"""
train_survival.py — XGBoost AFT survival model training

Trains on per-drive aggregates with real time-to-failure bounds, using a
shuffled, seeded 80/20 split for early stopping. A quality gate refuses to
export a model trained on too little data, so the continuous-training
pipeline can never overwrite production weights with a degenerate model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

FEATURES = ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"]
MIN_DRIVES = 1000
MIN_FAILURES = 10
SEED = 42

def compute_real_ttf(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes real Time-To-Failure bounds per drive from actual observation dates.
    
    For each unique serial_number:
      - ttf = number of days from first_seen to last_seen (the drive's real lifespan)
      - y_lower = ttf (the exact observed lifespan — a hard lower bound)
      - y_upper = ttf if failed==1, else inf (right-censored: drive still alive)
    
    This replaces synthetic math with ground-truth survival times from the dataset.
    """
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False))

    drive_stats = (
        df.group_by("serial_number")
        .agg([
            pl.col("date").min().alias("first_seen"),
            pl.col("date").max().alias("last_seen"),
            pl.col("failure").max().alias("ever_failed"),
            pl.col("smart_5_raw").mean().alias("smart_5_raw"),
            pl.col("smart_187_raw").mean().alias("smart_187_raw"),
            pl.col("smart_188_raw").mean().alias("smart_188_raw"),
            pl.col("smart_197_raw").mean().alias("smart_197_raw"),
            pl.col("smart_198_raw").mean().alias("smart_198_raw"),
        ])
    )

    drive_stats = drive_stats.with_columns([
        ((pl.col("last_seen") - pl.col("first_seen")).dt.total_days() + 1).alias("ttf")
    ])

    drive_stats = drive_stats.with_columns([
        pl.col("ttf").clip(lower_bound=1).alias("y_lower"),
        pl.when(pl.col("ever_failed") == 1)
          .then(pl.col("ttf").clip(lower_bound=1))
          .otherwise(float("inf"))
          .alias("y_upper"),
        pl.col("ever_failed").alias("failure"),
    ])

    print(f"  drives processed: {len(drive_stats)}")
    print(f"  failed drives:    {drive_stats['ever_failed'].sum()}")
    print(f"  median TTF days:  {drive_stats['ttf'].median():.1f}")

    return drive_stats

def check_data_quality(drive_stats: pl.DataFrame, min_drives: int, min_failures: int) -> None:
    n_drives = len(drive_stats)
    n_failed = int(drive_stats["ever_failed"].sum())
    if n_drives < min_drives or n_failed < min_failures:
        raise ValueError(
            f"Quality gate failed: {n_drives} drives / {n_failed} failures "
            f"(need >= {min_drives} drives and >= {min_failures} failures). Refusing to export."
        )


def shuffled_split(n: int, eval_fraction: float = 0.2, seed: int = SEED):
    """Seeded random split. group_by() output order is arbitrary, so slicing is not a valid split."""
    idx = np.random.default_rng(seed).permutation(n)
    n_eval = max(1, int(n * eval_fraction))
    return idx[n_eval:], idx[:n_eval]


def train_survival_model(data_path: Path, use_gpu: bool, output_path: Path | None = None,
                         min_drives: int = MIN_DRIVES, min_failures: int = MIN_FAILURES):
    print(f"loading data from {data_path}")
    df = pl.read_parquet(data_path)

    print("computing real per-drive TTF from observation dates...")
    drive_stats = compute_real_ttf(df)
    check_data_quality(drive_stats, min_drives, min_failures)

    X = drive_stats.select(FEATURES).to_numpy()
    y_lower = drive_stats["y_lower"].to_numpy()
    y_upper = drive_stats["y_upper"].to_numpy()

    # Shuffled 80/20 train/eval split for early stopping
    train_idx, eval_idx = shuffled_split(len(X))
    X_train, X_eval = X[train_idx], X[eval_idx]
    y_lower_train, y_lower_eval = y_lower[train_idx], y_lower[eval_idx]
    y_upper_train, y_upper_eval = y_upper[train_idx], y_upper[eval_idx]

    dtrain = xgb.DMatrix(X_train)
    dtrain.set_float_info('label_lower_bound', y_lower_train)
    dtrain.set_float_info('label_upper_bound', y_upper_train)

    deval = xgb.DMatrix(X_eval)
    deval.set_float_info('label_lower_bound', y_lower_eval)
    deval.set_float_info('label_upper_bound', y_upper_eval)
    
    params = {
        'objective': 'survival:aft',
        'eval_metric': 'aft-nloglik',
        'aft_loss_distribution': 'normal',
        'aft_loss_distribution_scale': 1.2,
        'tree_method': 'hist',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'seed': SEED,
    }
    
    if use_gpu:
        params['device'] = 'cuda'
        
    print("Training XGBoost AFT model (up to 300 rounds with early stopping)...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(deval, "eval")],
        early_stopping_rounds=20,
        verbose_eval=50
    )
    
    if output_path is None:
        project_root = data_path.parent.parent.parent
        output_path = project_root / "src" / "api" / "survival_model.json"

    # Write to a temp file then atomically replace, so a crash mid-write can
    # never leave a truncated model where the API will load it.
    print(f"Exporting XGBoost model to {output_path}...")
    tmp_path = output_path.with_name(output_path.name + ".tmp.json")
    bst.save_model(tmp_path)
    tmp_path.replace(output_path)
    print("Model saved successfully.")
    return bst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--min-drives", type=int, default=MIN_DRIVES)
    parser.add_argument("--min-failures", type=int, default=MIN_FAILURES)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "processed" / "survival_data.parquet"

    try:
        train_survival_model(data_path, use_gpu=not args.cpu,
                             min_drives=args.min_drives, min_failures=args.min_failures)
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
