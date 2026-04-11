"""
visualise.py
All charts and maps for the project.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
import os

os.makedirs('outputs', exist_ok=True)

BLUE   = '#2563EB'
TEAL   = '#0D9488'
RED    = '#DC2626'
GRAY   = '#6B7280'


def plot_scatter(df: pd.DataFrame, x: str, xlabel: str, title: str, filename: str):
    """Scatter plot with regression line and country labels for extremes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color points by region
    regions  = df['region'].fillna('Unknown').unique()
    palette  = dict(zip(regions, plt.cm.Set2.colors[:len(regions)]))
    colors   = df['region'].fillna('Unknown').map(palette)

    ax.scatter(df[x], df['log_gdp'], c=colors, alpha=0.7,
               edgecolors='white', linewidth=0.4, s=60)

    # Regression line
    m, b = np.polyfit(df[x].dropna(), df.loc[df[x].notna(), 'log_gdp'], 1)
    x_range = np.linspace(df[x].min(), df[x].max(), 200)
    ax.plot(x_range, m * x_range + b, color=RED,
            linewidth=1.5, linestyle='--', label=f'slope = {m:.3f}')

    # Label extreme outliers
    q_high = df['log_gdp'].quantile(0.92)
    q_low  = df['log_gdp'].quantile(0.08)
    for _, row in df.iterrows():
        if row['log_gdp'] > q_high or row['log_gdp'] < q_low:
            ax.annotate(
                row['country_name'],
                (row[x], row['log_gdp']),
                fontsize=7.5, alpha=0.8,
                xytext=(4, 2), textcoords='offset points'
            )

    # Legend for regions
    handles = [plt.Line2D([0],[0], marker='o', color='w',
                markerfacecolor=c, markersize=8, label=r)
               for r, c in palette.items()]
    ax.legend(handles=handles, title='Region',
              loc='upper right', fontsize=8, title_fontsize=9)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Log GDP per capita (PPP)', fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = f'outputs/{filename}'
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")


def plot_both_scatters(df: pd.DataFrame):
    """Side-by-side: temperature and latitude vs wealth."""
    plot_scatter(df, 'mean_temp',
                 'Mean annual temperature (°C)',
                 'Temperature vs Wealth — 160+ countries',
                 'temp_vs_gdp.png')

    plot_scatter(df, 'abs_latitude',
                 'Absolute latitude (degrees from equator)',
                 'Distance from Equator vs Wealth',
                 'lat_vs_gdp.png')


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix of the three main variables."""
    cols = ['log_gdp', 'mean_temp', 'abs_latitude']
    labels = ['Log GDP', 'Temperature', 'Abs latitude']

    corr = df[cols].corr()
    corr.index   = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                annot_kws={'size': 11}, ax=ax)
    ax.set_title('Correlation matrix', fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png', dpi=150)
    plt.show()
    print("Saved → outputs/correlation_heatmap.png")


def plot_world_map(df: pd.DataFrame):
    """Interactive Plotly choropleth — bubble = wealth, color = temperature."""
    fig = px.scatter_geo(
        df,
        locations='iso3',
        color='mean_temp',
        size='log_gdp',
        size_max=20,
        hover_name='country_name',
        hover_data={
            'mean_temp':    ':.1f',
            'gdp_ppp':      ':,.0f',
            'abs_latitude': ':.1f',
            'iso3':         False,
        },
        color_continuous_scale='RdYlBu_r',
        labels={
            'mean_temp':    'Avg temp (°C)',
            'gdp_ppp':      'GDP/capita PPP',
            'abs_latitude': 'Abs. latitude',
        },
        title='Climate and Wealth — bubble size = log GDP per capita',
        projection='natural earth',
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html('outputs/world_map.html')
    fig.show()
    print("Saved → outputs/world_map.html")


def plot_residuals(df: pd.DataFrame, model, title: str = 'Residuals'):
    """Plot actual vs predicted and highlight biggest deviations."""
    df = df.copy()
    df['predicted'] = model.fittedvalues
    df['residual']  = model.resid

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df['predicted'], df['residual'],
               alpha=0.6, color=BLUE, edgecolors='white', linewidth=0.3, s=50)
    ax.axhline(0, color=RED, linewidth=1, linestyle='--')

    # Label biggest residuals
    for _, row in df.nlargest(6, 'residual').iterrows():
        ax.annotate(row['country_name'],
                    (row['predicted'], row['residual']),
                    fontsize=7.5, color='green',
                    xytext=(4, 2), textcoords='offset points')
    for _, row in df.nsmallest(6, 'residual').iterrows():
        ax.annotate(row['country_name'],
                    (row['predicted'], row['residual']),
                    fontsize=7.5, color='red',
                    xytext=(4, -8), textcoords='offset points')

    ax.set_xlabel('Predicted log GDP', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('outputs/residuals.png', dpi=150)
    plt.show()
    print("Saved → outputs/residuals.png")