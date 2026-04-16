"""
main.py
Run the full analysis pipeline.
"""

from src.data_loader import build_dataset
from src.analysis    import run_all_models, get_outliers, summarize_by_region, model_comparison
from src.visualise   import (plot_both_scatters, plot_correlation_heatmap,
                              plot_world_map, plot_residuals,
                              plot_region_wealth_boxplot, plot_residuals_by_region)


def main():
    print("=" * 55)
    print("  Climate and Wealth — Analysis Pipeline")
    print("=" * 55)

    # ── 1. Build dataset ──────────────────────────────────────────
    df = build_dataset(temp_filepath='data/average-monthly-surface-temperature.csv')
    df.to_csv('data/merged_dataset.csv', index=False)
    print(f"\nSaved merged dataset → data/merged_dataset.csv")

    # ── 2. Regional summary ─────────────────────────────────────
    region_summary = summarize_by_region(df)
    region_summary.to_csv('outputs/regional_summary.csv')
    print("Saved regional summary → outputs/regional_summary.csv")

    # ── 3. EDA visualisations ─────────────────────────────────────
    print("\nGenerating visualisations...")
    plot_both_scatters(df)
    plot_correlation_heatmap(df)
    plot_region_wealth_boxplot(df)
    plot_world_map(df)

    # ── 4. Regression ─────────────────────────────────────────────
    print("\n\nRunning regression models...")
    models = run_all_models(df)
    comparison = model_comparison(models)
    comparison.to_csv('outputs/model_comparison.csv')
    print("Saved model comparison → outputs/model_comparison.csv")

    # ── 5. Outlier and residual analysis ──────────────────────────
    print("\n\nOutlier analysis (Model 1 — temperature only):")
    get_outliers(df, models['temp_only'])

    plot_residuals(df, models['temp_only'],
                   'Residuals — Temperature model')
    plot_residuals_by_region(df, models['temp_only'])

    print("\n\nDone. Check outputs/ folder for all charts.")


if __name__ == '__main__':
    main()