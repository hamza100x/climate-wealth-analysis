"""
analysis.py
Regression models and statistical analysis.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def run_model(df: pd.DataFrame, formula: str, name: str):
    """Fit OLS model and print a clean summary."""
    model = smf.ols(formula, data=df).fit()
    rmse = float(np.sqrt(np.mean(np.square(model.resid))))
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  R² = {model.rsquared:.3f}   Adj R² = {model.rsquared_adj:.3f}"
          f"   N = {int(model.nobs)}")
    print(f"  AIC = {model.aic:.1f}   BIC = {model.bic:.1f}   RMSE = {rmse:.3f}")
    print(f"{'─'*50}")
    summary = model.params.to_frame('Coeff').join(
               model.pvalues.to_frame('p-value'))
    summary['Significant'] = summary['p-value'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    )
    print(summary.round(4).to_string())
    return model


def model_comparison(models: dict) -> pd.DataFrame:
    """Create a compact comparison table for fitted models."""
    rows = []
    for name, model in models.items():
        rmse = float(np.sqrt(np.mean(np.square(model.resid))))
        rows.append({
            'model': name,
            'R2': model.rsquared,
            'Adj_R2': model.rsquared_adj,
            'AIC': model.aic,
            'BIC': model.bic,
            'RMSE': rmse,
            'N': int(model.nobs),
        })

    table = pd.DataFrame(rows).set_index('model')
    print("\nModel comparison table:")
    print(table.round(4).to_string())
    return table


def summarize_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize wealth and climate patterns by world region."""
    summary = (
        df.groupby('region', dropna=False)
        .agg(
            countries=('country_name', 'nunique'),
            mean_gdp=('gdp_ppp', 'mean'),
            median_gdp=('gdp_ppp', 'median'),
            mean_temp=('mean_temp', 'mean'),
            mean_abs_latitude=('abs_latitude', 'mean'),
        )
        .sort_values('median_gdp', ascending=False)
    )

    print("\nRegional summary:")
    print(summary.round(2).to_string())
    return summary


def run_all_models(df: pd.DataFrame) -> dict:
    """Run all three regression models and return results."""

    # Correlation between the two predictors — explains multicollinearity
    corr = df[['mean_temp', 'abs_latitude']].corr().iloc[0, 1]
    print(f"\nCorrelation between temp and abs_latitude: {corr:.3f}")
    print("(explains why Model 3 barely improves over Model 2)\n")

    models = {}

    models['temp_only'] = run_model(
        df, 'log_gdp ~ mean_temp',
        'Model 1 — Temperature only'
    )
    models['lat_only'] = run_model(
        df, 'log_gdp ~ abs_latitude',
        'Model 2 — Latitude only'
    )
    models['both'] = run_model(
        df, 'log_gdp ~ mean_temp + abs_latitude',
        'Model 3 — Temperature + Latitude (combined)'
    )

    return models


def get_outliers(df: pd.DataFrame, model, n: int = 8) -> pd.DataFrame:
    """Return countries that deviate most from model predictions."""
    df = df.copy()
    df['predicted'] = model.fittedvalues
    df['residual']  = model.resid
    df['std_residual'] = model.get_influence().resid_studentized_internal

    cols = ['country_name', 'region', 'gdp_ppp', 'mean_temp',
            'abs_latitude', 'predicted', 'residual', 'std_residual']

    richer = df.nlargest(n, 'std_residual')[cols]
    poorer = df.nsmallest(n, 'std_residual')[cols]

    print("\nRicher than temperature predicts:")
    print(richer[['country_name', 'gdp_ppp', 'mean_temp', 'residual', 'std_residual']].to_string(index=False))

    print("\nPoorer than temperature predicts:")
    print(poorer[['country_name', 'gdp_ppp', 'mean_temp', 'residual', 'std_residual']].to_string(index=False))

    return richer, poorer