# KHS Dissertation — Working Roadmap
### Modelling Housing-Based Financial Vulnerability and Insurance Risk Among Kenyan Households
**MSc Data Science & Analytics · Strathmore University**  
**Data: 2023/24 Kenya Housing Survey (KNBS) · 21,347 households · 47 counties**

---

## Confirmed thesis direction

> Construct a multi-dimensional Housing Financial Vulnerability Score (HFVS) from KHS microdata,
> model its determinants using machine learning, and validate county-level risk gradients
> against IRA 2025 loss ratios as external ground truth.

No insurance variable exists in the data — this is the correct framing.
You are doing what actuaries do: inferring risk from observable household characteristics.

---

## The four risk dimensions (confirmed from codebook)

| Dimension | Key columns | What it measures |
|---|---|---|
| Financial stress | `c14_1`, `k05`, `c14_2`, `c14_3`, `d20__1–8` | Rent burden, savings capacity, barriers |
| Tenure insecurity | `k35`, `k29`, `k34`, `l26`, `i00`, `k02` | Eviction risk, land ownership, legal protection |
| Physical hazard | `e06`, `e07`, `e08`, `e09__1–13` | Flood, mudslide, hazard proximity (enumerator-observed) |
| Financial exclusion | `d20__4`, `k27`, `l01_2`, `l02__1–8` | Loan access, financing formality |
| Dwelling quality | Dwelling Units file (not yet loaded) | Wall, floor, roof construction materials |

---

## Notebook execution plan

### `01_data_understanding.ipynb` — Week 1
**Goal:** Know every variable that matters before touching a model.

Steps:
1. Load household parquet + dwelling units parquet
2. Run full null audit on both — note structural vs random missingness
3. Load value labels JSON, create a lookup function for readable categories
4. Profile the four risk dimension columns individually
5. Check urban/rural distributions per county (a07_1 × countycode)
6. Load and inspect Housing Mortgage + Loan files — check join rates to household

Deliverable: A printed summary table of usable columns per dimension + join rate report.

---

### `02_feature_engineering.ipynb` — Week 1–2
**Goal:** Turn raw survey columns into model-ready features and construct the HFVS.

#### Step 1 — Rent burden ratio
```python
# c14_1 = monthly expenditure, k05 = monthly rent
df = df.with_columns([
    (pl.col("k05") / pl.col("c14_1")).alias("rent_burden_ratio")
])
# Flag: rent_burden > 0.30 = housing-stressed (standard threshold)
df = df.with_columns([
    (pl.col("rent_burden_ratio") > 0.30).cast(pl.Int8).alias("rent_stressed")
])
```

#### Step 2 — Tenure insecurity score (additive)
```python
df = df.with_columns([
    (
        pl.col("k35").cast(pl.Int8) +          # eviction threat
        (pl.col("k02") == 0).cast(pl.Int8) +   # no written agreement
        pl.col("k29").cast(pl.Int8) +           # had rent dispute
        (pl.col("i00") == 0).cast(pl.Int8)      # does not own land
    ).alias("tenure_insecurity_score")
])
```

#### Step 3 — Physical hazard score
```python
hazard_cols = ["e06", "e07"] + [f"e09__{i}" for i in range(1, 14)]
df = df.with_columns([
    pl.sum_horizontal([pl.col(c).cast(pl.Int8) for c in hazard_cols if c in df.columns])
      .alias("hazard_score")
])
```

#### Step 4 — Financial exclusion score
```python
df = df.with_columns([
    (
        pl.col("d20__4").cast(pl.Int8) +       # doesn't qualify for loan
        (pl.col("k27").is_null()).cast(pl.Int8) # no loan access reported
    ).alias("financial_exclusion_score")
])
```

#### Step 5 — Utility deprivation index
Map c08 (no electricity), c01_1 (unimproved water), c04 (unimproved sanitation)
to binary flags, sum into deprivation_index.

#### Step 6 — Composite HFVS
```python
# Normalise each sub-score 0–1, then weighted average
# Weights are provisional — refine with PCA or expert judgment
df = df.with_columns([
    (
        0.30 * pl.col("rent_burden_norm") +
        0.25 * pl.col("tenure_insecurity_norm") +
        0.25 * pl.col("hazard_norm") +
        0.20 * pl.col("deprivation_norm")
    ).alias("hfvs")
])
```

Deliverable: Master analytical dataset saved as `master_hfvs.parquet` on Drive.

---

### `03_model_training.ipynb` — Week 2–3
**Goal:** Model HFVS and its binary threshold version. Explain via SHAP.

#### Models to run (in order)
1. Logistic regression on `hfvs_high` (binary, HFVS > 0.60) — interpretable baseline
2. XGBoost on binary target — primary model
3. XGBoost regression on continuous HFVS — secondary
4. Random Forest — for ensemble/comparison

#### Features to include
- County (encoded), urban/rural (`a07_1`)
- Household size (`a12`)
- Education level (from Individual file after aggregation)
- Employment rate (from Individual file)
- Monthly expenditure (`c14_1`), savings (`c14_2`)
- Dwelling materials (from Dwelling Units file)
- All four risk dimension sub-scores

#### Evaluation
- Classification: AUC-ROC, precision-recall, F1 by county
- Regression: RMSE, R², spatial residual plot
- Explainability: SHAP summary plot, county-level SHAP beeswarm

```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

Deliverable: Trained models + SHAP plots saved to outputs/figures/

---

### `04_county_risk_mapping.ipynb` — Week 3–4
**Goal:** Aggregate to county level, build risk map, validate against IRA.

#### County aggregation
```python
county_risk = (
    master
    .group_by("countycode")
    .agg([
        pl.col("hfvs").mean().alias("mean_hfvs"),
        pl.col("hfvs").quantile(0.75).alias("p75_hfvs"),
        pl.col("rent_stressed").mean().alias("pct_rent_stressed"),
        pl.col("tenure_insecurity_score").mean().alias("mean_tenure_risk"),
        pl.col("hazard_score").mean().alias("mean_hazard"),
        pl.col("hhweight").sum().alias("total_weight"),
        pl.len().alias("n_households")
    ])
    .join(county_names, on="countycode")  # add county name
    .sort("mean_hfvs", descending=True)
)
```

#### IRA validation
- Download IRA Insurance Annual Report 2025 — county-level loss ratios
- Merge on county name
- Run Pearson + Spearman correlation: `mean_hfvs` vs `loss_ratio`
- Run OLS regression: loss_ratio ~ mean_hfvs + pct_urban + mean_income_proxy
- Expected finding: counties with high HFVS should have higher loss ratios
- This is your Chapter 5 validation argument

#### Choropleth map
Use Kenya county shapefile (available from KNBS GitHub) + geopandas + matplotlib
to produce a publication-quality county risk map.

```python
import geopandas as gpd
kenya = gpd.read_file("kenya_counties.shp")
kenya = kenya.merge(county_risk, left_on="COUNTY_COD", right_on="countycode")
kenya.plot(column="mean_hfvs", cmap="RdYlGn_r", legend=True)
```

Deliverable: County risk table CSV + choropleth map PNG

---

## File loading order (always in this sequence)

```
1. Household Information     ← spine of everything
2. Dwelling Units            ← join on interview__key (dwelling quality)
3. Individual Microdata      ← aggregate first, then join (education, employment)
4. Housing Mortgage          ← join on interview__key (flag mortgage households)
5. Housing Loan              ← join on interview__key (flag loan households)
6. County Microdata          ← join on countycode (county-level context)
```

Files NOT needed for core dissertation:
- Water Service Providers, NEMA, Institution, Project Info, Real Estate, Land Parcels
  (useful for supplementary analysis only — skip for now)

---

## Data issues to watch for

| Issue | Where | Fix |
|---|---|---|
| `k05`, `c14_1` heavily skewed | Financial columns | Winsorise at 99th percentile |
| k-section columns 97%+ null | Renters only (skip logic) | Separate renter vs owner analysis |
| Urban coded as 1/2 not string | `a07_1` | Map: 1=Rural, 2=Urban from value labels |
| `countycode` is string "07" not int | Joins | Cast to int or keep as string consistently |
| High null rate in l-section | Owners only | Expected — structural missingness |

---

## Immediate next session (open Colab now)

```
[ ] Load dwelling units parquet
[ ] Run null audit on dwelling units
[ ] Check join rate: dwelling units → household on interview__key
[ ] Extract variable labels from dwelling units DTA (same codebook extraction code)
[ ] Profile wall material, floor material, roof material columns
[ ] Compute first rent burden ratio from c14_1 and k05
[ ] Check how many households have both c14_1 and k05 non-null
```

---

## GitHub commit schedule

| When | What to commit |
|---|---|
| End of each notebook | Notebook + any new src/ functions |
| Never | .parquet, .dta, .csv, .env files |
| After notebook 02 | `src/features.py` with all HFVS functions |
| After notebook 03 | `src/models.py` with training pipeline |
| After notebook 04 | `outputs/figures/` choropleth + SHAP plots |

Push command after each session:
```bash
# In Colab terminal or !cell
!git add notebooks/ src/ outputs/figures/
!git commit -m "feat: [describe what you did]"
!git push origin main
```

---

## Dissertation chapter map

| Chapter | Title | Notebook |
|---|---|---|
| 1 | Introduction + Kenya housing context | — |
| 2 | Literature review — housing risk, InsurTech, actuarial ML | — |
| 3 | Methodology — CRISP-DM, feature engineering, XGBoost | 01 + 02 |
| 4 | Results — HFVS distribution, model performance, SHAP | 03 |
| 5 | Validation + policy implications — IRA correlation, county map | 04 |
| 6 | Conclusion + limitations | — |

---

*Last updated: April 2026*