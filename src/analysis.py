"""
analysis.py
Regression models and statistical analysis.
"""

import pandas as pd
import statsmodels.formula.api as smf


def run_model(df: pd.DataFrame, formula: str, name: str):
    """Fit OLS model and print a clean summary."""
    model = smf.ols(formula, data=df).fit()
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  R² = {model.rsquared:.3f}   Adj R² = {model.rsquared_adj:.3f}"
          f"   N = {int(model.nobs)}")
    print(f"{'─'*50}")
    summary = model.params.to_frame('Coeff').join(
               model.pvalues.to_frame('p-value'))
    summary['Significant'] = summary['p-value'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    )
    print(summary.round(4).to_string())
    return model


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

    cols = ['country_name', 'region', 'gdp_ppp', 'mean_temp',
            'abs_latitude', 'predicted', 'residual']

    richer = df.nlargest(n, 'residual')[cols]
    poorer = df.nsmallest(n, 'residual')[cols]

    print("\nRicher than temperature predicts:")
    print(richer[['country_name', 'gdp_ppp', 'mean_temp', 'residual']].to_string(index=False))

    print("\nPoorer than temperature predicts:")
    print(poorer[['country_name', 'gdp_ppp', 'mean_temp', 'residual']].to_string(index=False))

    return richer, poorer