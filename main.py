"""
main.py
Run the full analysis pipeline.
"""

from src.data_loader import build_dataset
from src.analysis    import run_all_models, get_outliers
from src.visualise   import (plot_both_scatters, plot_correlation_heatmap,
                              plot_world_map, plot_residuals)


def main():
    print("=" * 55)
    print("  Climate and Wealth — Analysis Pipeline")
    print("=" * 55)

    # ── 1. Build dataset ──────────────────────────────────────────
    df = build_dataset(temp_filepath='data/average-monthly-surface-temperature.csv')
    df.to_csv('data/merged_dataset.csv', index=False)
    print(f"\nSaved merged dataset → data/merged_dataset.csv")

    # ── 2. EDA visualisations ─────────────────────────────────────
    print("\nGenerating visualisations...")
    plot_both_scatters(df)
    plot_correlation_heatmap(df)
    plot_world_map(df)

    # ── 3. Regression ─────────────────────────────────────────────
    print("\n\nRunning regression models...")
    models = run_all_models(df)

    # ── 4. Outlier analysis ───────────────────────────────────────
    print("\n\nOutlier analysis (Model 1 — temperature only):")
    get_outliers(df, models['temp_only'])

    plot_residuals(df, models['temp_only'],
                   'Residuals — Temperature model')

    print("\n\nDone. Check outputs/ folder for all charts.")


if __name__ == '__main__':
    main()