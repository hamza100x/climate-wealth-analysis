# Climate and Wealth Analysis

A data analysis pipeline that explores how climate variables (temperature and latitude) relate to GDP per capita across countries.
## Dataset sources

| Variable | Source | Link |
|---|---|---|
| GDP per capita PPP | World Bank via `wbdata` | [data.worldbank.org](https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.CD) |
| Mean annual temperature | World Bank Climate Portal | [climateknowledgeportal.worldbank.org](https://climateknowledgeportal.worldbank.org) |
| Country latitude | ISO 3166 country list | [GitHub — lukes/ISO-3166](https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes) |


The project:
- builds a merged country-level dataset from local CSV files,
- runs linear regression models,
- generates visualizations,
- compares model fit with R², AIC, BIC, and RMSE,
- summarizes GDP patterns by region,
- highlights countries that overperform or underperform relative to climate-based predictions.

## What this project does

1. Loads and normalizes GDP, temperature, coordinates, and region data.
2. Engineers analysis features:
- `log_gdp` = natural log of GDP per capita
- `abs_latitude` = absolute distance from equator
3. Fits three OLS models:
- `log_gdp ~ mean_temp`
- `log_gdp ~ abs_latitude`
- `log_gdp ~ mean_temp + abs_latitude`
4. Produces charts and an interactive world map.
5. Prints outlier countries using model residuals.
6. Adds region-level summaries and residual diagnostics to show where the pooled model fits poorly.

## Project structure

```text
climate-wealth-analysis/
  data/
    average-monthly-surface-temperature.csv
    countries-with-latitude.csv
    gdp-per-capita-worldbank.csv
    merged_dataset.csv           # generated
  outputs/
    temp_vs_gdp.png             # generated
    lat_vs_gdp.png              # generated
    correlation_heatmap.png     # generated
    region_wealth_boxplot.png   # generated
    residuals.png               # generated
    residuals_by_region.png     # generated
    world_map.html              # generated
    regional_summary.csv        # generated
    model_comparison.csv        # generated
  src/
    data_loader.py
    analysis.py
    visualise.py
  main.py
  requirements.txt
```

## Data inputs

Expected local files in `data/`:
- `gdp-per-capita-worldbank.csv`
- `average-monthly-surface-temperature.csv`
- `countries-with-latitude.csv`

Notes:
- GDP data is filtered to `year=2019` when available.
- Temperature is aggregated from monthly country records into one mean value per country.
- Country merges prioritize ISO3 codes when available.

## Setup

### 1) Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## Run the pipeline

```powershell
python main.py
```

The script will:
- build and save `data/merged_dataset.csv`,
- save regional summaries and model comparisons into `outputs/`,
- generate charts in `outputs/`,
- print regression summaries and outlier tables to the console.

## Output files

After a successful run:
- `data/merged_dataset.csv`
- `outputs/regional_summary.csv`
- `outputs/model_comparison.csv`
- `outputs/temp_vs_gdp.png`
- `outputs/lat_vs_gdp.png`
- `outputs/correlation_heatmap.png`
- `outputs/region_wealth_boxplot.png`
- `outputs/residuals.png`
- `outputs/residuals_by_region.png`
- `outputs/world_map.html`

## Key findings

**1. Temperature explains 24% of income variance, latitude explains 40%.**
Both climate variables correlate with national wealth, but latitude is a 
stronger predictor — countries farther from the equator tend to be richer.

**2. The two variables carry largely the same information.**
Their mutual correlation is ≈ −0.87 (cold countries are far from the equator,
by definition). Adding temperature after latitude raises R² only from 0.404 
to 0.421 — a 1.7 percentage point gain, suggesting minimal independent 
information.

**3. The sign flip reveals multicollinearity.**
Temperature is negative in isolation (−0.066, hotter = poorer) but turns 
positive in the combined model (+0.035). This is a suppressor effect caused 
by the high correlation between the two predictors — a statistical artefact,
not a change in the real-world relationship.

**4. Outliers point to omitted variables.**
Gulf states (Qatar, UAE, Saudi Arabia) are far wealthier than climate 
predicts → oil wealth. Cold countries like Tajikistan and Lesotho are far 
poorer than expected → landlocked geography, institutional quality, 
colonial history. These residuals are an argument for extending the model.

**5. Climate explains ~40% of income variation — the other 60% lies elsewhere.**
The strongest next candidates are institutional quality (rule of law),
colonial history, and natural resource endowments. These are the variables
Acemoglu & Robinson argue dominate once geography is controlled for.

**6. The regional gap is large.**
In the current dataset, Europe has by far the highest median GDP per capita,
while Africa sits at the bottom. The regional residual plots show that the
pooled climate model still leaves a lot of structure unexplained, which is a
hint that geography is only one piece of the story.
### Model comparison

| Model | Formula | $R^2$ | Key coefficient takeaway |
|---|---|---:|---|
| Model 1 | `log_gdp ~ mean_temp` | 0.241 | Temperature is negative ($-0.0655$, $p < 0.001$) |
| Model 2 | `log_gdp ~ abs_latitude` | 0.404 | Latitude is positive ($+0.0421$, $p < 0.001$) |
| Model 3 | `log_gdp ~ mean_temp + abs_latitude` | 0.421 | Latitude stays strong; temperature is smaller and positive ($p = 0.0342$) |

### Residual highlights (temperature-only model)

- Richer than predicted: Qatar, Cayman Islands, United Arab Emirates, Saudi Arabia.
- Poorer than predicted: Tajikistan, Burundi, Lesotho, Central African Republic.

Conclusion: climate variables are meaningfully associated with income differences, but most variation still comes from non-climate factors. These results are correlational and should not be interpreted as causal.

## Model interpretation quick guide

- A negative temperature coefficient in Model 1 suggests hotter countries tend to have lower GDP per capita (on average).
- A positive absolute-latitude coefficient in Model 2 suggests countries farther from the equator tend to be richer (on average).
- Model 3 tests both effects jointly.

These are correlations, not causal claims.

## Troubleshooting

- `FileNotFoundError` for data files:
  - confirm the three input CSVs exist under `data/` with the expected names.
- Column mapping errors:
  - ensure GDP includes country, ISO3 code, and GDP per capita columns.
  - ensure temperature includes country and temperature-value columns.
- Plot windows not showing:
  - run in a local Python environment with GUI support, or inspect saved files in `outputs/`.

## Future improvements

- Add tests for schema validation and transforms.
- Add command-line options for year and input paths.
- Add notebook-based exploratory analysis and report export.
