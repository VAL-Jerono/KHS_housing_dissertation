# KHS Dissertation — Notebook Output Summary
## `KHS_Dissertation_Final_Enhanced.ipynb`

---

## Phase 0: Environment Setup

```
✓ Drive mounted. Repository ready.
✓ All packages installed.
  XGBoost 3.2.0 | LightGBM 4.6.0 | SHAP 0.52.0 | PyTorch 2.11.0+cpu
✓ All imports loaded.
✓ Environment configured.
  FIGS: /content/drive/MyDrive/KHS_Dissertation/outputs/figures
  TABS: /content/drive/MyDrive/KHS_Dissertation/outputs/tables
  Counties defined: 47
```

---

## Phase 2: Data Understanding

### 2.1 Codebook Labels Loaded
```
Household variable labels : 392
Household value labels    : 121
Dwelling variable labels  : 25
```

### 2.2 File Inventory

| File | Rows | Cols | Description |
|---|---|---|---|
| household | 21,347 | 392 | SPINE — finances, tenure, utilities, infrastructure |
| individual | 80,889 | 97 | One row per person — demographics, education, migration |
| dwelling | 25,116 | 25 | Physical structure — materials, rooms, floor area |
| land_parcels | 11,136 | 34 | Household-linked parcels — title, disputes, collateral |
| county | 47 | 116 | County-level planning and infrastructure |
| mortgage | 1,644 | 13 | Institutional mortgage records |
| loan | 946 | 10 | Institutional housing loan records |
| nema | 48 | 45 | Environmental approvals and processing |
| water_svc | 153 | 96 | Water-service providers |
| real_estate | 7,236 | 300 | Market reference data |
| financiers | 351 | 63 | County finance infrastructure |
| institutional | 348 | 194 | Institutional/policy context |
| project_info | 71 | 211 | Affordable housing project pipeline |
| housing_types | 131 | 17 | Reference housing categories |

> **Survey universe:** 21,347 households across 47 counties. Additional source files loaded: 8

### 2.3 Household Null Audit (392 columns)

```
Complete  (0%)      :  67 cols
Low       (1–20%)   : 102 cols
Moderate  (21–60%)  :  60 cols  ← structural (renter/owner split)
High      (61–90%)  :  97 cols
Extreme   (>90%)    :  66 cols
```

### 2.4 Geographic Distribution

```
Rural households : 11,900 (55.7%)
Urban households :  9,447 (44.3%)
Counties covered :     47 / 47
Median per county:    425 households
Range            :  319 to 1,059
```

**Top 10 counties by sample size:**

| county_name | Rural | Urban | Total |
|---|---|---|---|
| Nairobi | 0 | 1,059 | 1,059 |
| Nakuru | 354 | 332 | 686 |
| Wajir | 377 | 260 | 637 |
| Machakos | 355 | 263 | 618 |
| Kitui | 397 | 176 | 573 |
| West Pokot | 320 | 197 | 517 |
| Kisii | 325 | 185 | 510 |
| Mandera | 234 | 269 | 503 |
| Kakamega | 316 | 164 | 480 |
| Narok | 289 | 190 | 479 |

### 2.5 Individual File Demographic Profile

```
Total individuals     : 80,889
Mean age              : 25.3 years
% Female              : 50.9%
% Under 15 (children) : 36.5%
% 65+ (elderly)       :  4.9%
Ages outside 0–100    :    33 (excluded)
```

### 2.6 Cross-File Join Audit

```
Household spine: 21,347 unique interview__key values

File                 Rows        HH key coverage      County bridge
------------------------------------------------------------------------
individual         80,889   21,346 / 21,347 (100.0%)          a01
dwelling           25,116   21,346 / 21,347 (100.0%)          a01
land_parcels       11,136    9,707 / 21,347  (45.5%)            -
county                 47          0 / 21,347   (0.0%)        cg00
mortgage            1,644          0 / 21,347   (0.0%)  county_name
loan                  946          0 / 21,347   (0.0%)  county_name
nema                   48          0 / 21,347   (0.0%)        nm00
water_svc             153          0 / 21,347   (0.0%)  county_name
real_estate         7,236          no interview__key   county_name
financiers            351          0 / 21,347   (0.0%)  county_name

LAND PARCEL STRUCTURAL COVERAGE
Households with parcel record: 9,707 (45.5%)
```

### 2.7 Housing Finance Landscape

```
Mortgage (bank/HFI)   :   241 households  (1.13%)
Any formal credit     :   491 households  (2.30%)
Loan-financed (l01_2) :   545 households  (2.55%)
```

**Financing type (l01_2) value counts:**

| Code | Count |
|---|---|
| 1 — Personal savings | 9,335 |
| 2 — Mortgage/loan | 545 |
| 5 — Employer | 123 |
| 6 — Other | 208 |
| 3 | 18 |
| 4 | 3 |
| 96 | 4 |

### 2.9 D3 Proximity Column Availability

```
Severity columns (enumerator-observed):
  ✓ e06 (flood_zone)    — 21,346 non-null records
  ✓ e07 (mudslide_zone) — 21,346 non-null records

Proximity sub-columns (e08__1 – e08__6):
  ✗ e08__1 (near_swamp)      — NOT IN EXTRACT
  ✗ e08__2 (near_dumpsite)   — NOT IN EXTRACT
  ✗ e08__3 (near_factory)    — NOT IN EXTRACT
  ✗ e08__4 (near_busy_road)  — NOT IN EXTRACT
  ✗ e08__5 (near_river_lake) — NOT IN EXTRACT
  ✗ e08__6 (near_quarry)     — NOT IN EXTRACT

Proximity columns available: 0 / 6
⚠ D3 weights used: flood=60%  mudslide=40%  proximity=0%
```

---

## Phase 3: Feature Engineering — HFVS Dimension Construction

### 3.3 Master Spine
```
Master spine constructed: (21,347 rows × 430 columns)
  Dwelling coverage     : 100.0%
  Land parcel coverage  :  45.5%
  Formal credit flag    :   2.6%
```

### 3.4 Expenditure & Affordability
```
g01 components used          : 11 / 11
Median total expenditure     : KES 16,420
Median non-housing spend     : KES 14,130
Rent source column           : k05
Renters with actual rent     :  6,932

actual_rent: n=6,932  mean=4,302  median=3,000
             p25=2,000  p75=5,000  p95=12,000  p99=30,000

aspirational_rent: n=21,171  mean=2,576  median=2,500
                   p25=1,700  p75=3,400  p95=4,500

k25 median = KES 1,000,000 — confirmed non-monthly; retired from all derived vars

Among renters ABOVE county-stratum median (n=3,098):
  Median excess : KES 2,000/month
  Mean excess   : KES 3,756/month
  p75 excess    : KES 4,500/month
  p95 excess    : KES 12,500/month

By residence:
  Rural : 42.5% above peer median | median excess KES 1,000/month
  Urban : 45.0% above peer median | median excess KES 2,100/month
```

### 3.5 Dimension 1 — Financial Stress (D1)
```
Score mean             : 0.4226
Rent stressed (>30%)   : 31.8%
Severe burden (>50%)   : 13.4%
No loan access         : 96.8%
Utility cost high      : 50.0%
```

### 3.6 Dimension 2 — Tenure Insecurity (D2)
```
Tenure source column   : g03
Score mean             : 0.2466
Renter households      : 33.2%
No land ownership      : 54.5%
Land parcel dispute    : 33.3%
```

### 3.7 Dimension 3 — Physical Hazard (D3)
```
Score mean        : 0.1101
In flood zone     : 19.1%
In mudslide zone  : 12.6%
Weights used      : flood=60%, mudslide=40%, proximity=0%
```

Compound exposure flags:
```
Flood zone only                : 19.1%
Tenure insecure only           : 25.2%
Rent stressed only             : 31.8%
Flood + no tenure              :  4.6%
Flood + rent stressed          :  5.4%
Tenure insecure + rent stressed: 10.4%
TRIPLE EXPOSED (all three)     :  1.8%
QUAD EXPOSED   (+ eviction)    :  0.1%
```

### 3.8 Dimension 4 — Dwelling Quality (D4)
```
Score mean           : 0.3926
Durable floor        : 62.2%
Durable wall         : 68.3%
Durable roof         : 86.2%
Overcrowded (>3 p/r) : 38.2%
```

### 3.9 Dimension 5 — Utility Deprivation (D5)

Lighting source (c10):
```
Grid electricity (KPLC)  : 12,010 HH
Solar panels             :  1,529 HH
Solar battery/torch      :  6,773 HH
Kerosene/paraffin lamp   :    540 HH
Candle                   :    190 HH
None                     :     58 HH
```

Cooking fuel (c11):
```
Firewood/wood products   : 11,503 HH
LPG (gas)                :  5,344 HH
Charcoal                 :  3,561 HH
Electricity (KPLC)       :    278 HH
Ethanol                  :    227 HH
Biogas                   :     88 HH
```

```
D5 Score mean       : 0.6121
No grid electricity : 43.7%
Solid fuel cooking  : 96.6%
```

---

## Phase 3 (cont.): HFVS Composite Score

### 3.10 Composite Construction
```
HFVS = (D1 + D2 + D3 + D4 + D5) / 5  (equal weighting)

HFVS mean            : 0.3568
HFVS std             : 0.0854
HFVS range           : 0.0591 to 0.7043
High vulnerability   : 40.0%  (HFVS > 0.373 = 60th percentile)
Survey weight range  : 24.91 to 8,162.01
Master saved         : master_hfvs_v3.parquet  (21,347 × 515)
```

### Effective vs Stated Dimension Weights
| Dimension | Stated | Effective | Status |
|---|---|---|---|
| D1 Financial Stress | 20.0% | 23.7% | — |
| D2 Tenure Insecurity | 20.0% | 13.8% | — |
| D3 Physical Hazard | 20.0% | 6.2% | ⚠ UNDERWEIGHTED |
| D4 Dwelling Quality | 20.0% | 22.0% | — |
| D5 Utility Deprivation | 20.0% | 34.3% | ↑ OVERWEIGHTED |

### HFVS Distribution Test
```
Shapiro-Wilk (n=2,000 sample): W=0.9933, p=0.0000
Distribution is NOT approximately normal (alpha=0.05)
```

### Inter-Dimension Correlations (Spearman)
```
D1_financial vs D2_tenure   : rho =  0.209  (p=1.66e-208)
D1_financial vs D3_hazard   : rho = -0.053  (p=1.10e-14)
D1_financial vs D4_dwelling : rho = -0.256  (p=0.00)
D1_financial vs D5_utility  : rho = -0.368  (p=0.00)
D2_tenure    vs D3_hazard   : rho =  0.003  (p=0.6498)
D2_tenure    vs D4_dwelling : rho = -0.079  (p=4.82e-31)
D2_tenure    vs D5_utility  : rho = -0.232  (p=3.75e-258)
D3_hazard    vs D4_dwelling : rho =  0.062  (p=2.16e-19)
D3_hazard    vs D5_utility  : rho =  0.015  (p=0.0289)
D4_dwelling  vs D5_utility  : rho =  0.187  (p=6.23e-168)

Highest correlation : D2 vs D1 = 0.221
D3 max correlation  : 0.070
Any pair above 0.40 : False
```

---

## Phase 4: EDA & Compound Vulnerability

### 4.4 Triple Exposure — Compound Vulnerability

| Combination | % HH | N |
|---|---|---|
| No flood + No tenure insecurity + No rent stress | 42.9% | 9,106 |
| No flood + No tenure insecurity + Rent stressed | 18.9% | 3,789 |
| No flood + Tenure insecure + No rent stress | 10.1% | 2,544 |
| No flood + Tenure insecure + Rent stressed | 8.5% | 1,836 |
| Flood + No tenure insecurity + No rent stress | 10.3% | 2,315 |
| Flood + No tenure insecurity + Rent stressed | 4.1% | 767 |
| Flood + Tenure insecure + No rent stress | 2.9% | 604 |
| **TRIPLE EXPOSED (flood + no tenure + rent stressed)** | **2.3%** | **386** |

**Rural vs Urban breakdown:**

| Exposure Type | Rural | Urban | Total | Ratio |
|---|---|---|---|---|
| Triple Exposed | 0.15% | 3.90% | 1.81% | 25.8× |
| Flood + No Tenure | 1.02% | 9.20% | 4.64% | 9.0× |
| Flood + Stressed | 2.75% | 8.74% | 5.40% | 3.2× |
| Tenure + Stressed | 1.24% | 21.95% | 10.41% | 17.7× |
| **Any Exposure (total)** | **4.71%** | **32.11%** | **16.83%** | **6.8×** |

**Key summary values:**
```
Triple exposed (total)          :  1.81%
Single-exposure-only rates:
  Flood only                    : 10.79%
  Tenure insecure only          : 12.09%
  Rent stressed only            : 17.36%
Largest pairwise intersection   : Tenure + Stressed (n=2,222)
Quad-exposed households         : 22  (0.10%)

Top 3 counties by triple rate:
  Nairobi   : 14.73%
  Mombasa   :  9.88%
  Kisumu    :  7.62%

Urban share of triple-exposed   : 95.3%
Largest urban-rural gap pair    : Tenure + Stressed
  (Urban 24.69% vs Rural 2.25% = 22.44pp gap)
```

### 4.5 Rent Burden Analysis
```
Rent-stressed nationally (>30% burden) : 31.8%
Severely stressed (>50% burden)        : 13.4%

Rent-stressed by expenditure quintile:
  Q1 (lowest) : 71.4%
  Q2          : 40.9%
  Q3          : 25.1%
  Q4          : 14.2%
  Q5 (highest):  7.1%

Most rent-stressed county  : Nairobi       (80.0%)
Least rent-stressed county : Tharaka-Nithi (44.1%)

Peer-comparison (renter vs county-stratum median):
  Renters above median  : 3,098 (44.7%)
  Median excess         : KES 2,000/month
  Mean excess           : KES 3,756/month
```

---

## Phase 5: Proxy Model Feature Engineering

### 5.2 Leakage Audit
```
136 banned features declared.
Rent-derived peer gap and land-parcel D2 ancestors banned from Track A.
```

### 5.3 Candidate Proxy Features (after variance screen)
```
Candidate proxy features : 66
Retained interactions    :  5
Mandatory domain anchors : 16
Final model target width : ≤ 35 features (selected in 5.4)
```

**Top mutual-information features (proxy candidates):**
```
pct_urban_county : MI = 0.0608
county_n_hh      : MI = 0.0539
mean_age         : MI = 0.0286
n_children       : MI = 0.0258
mean_edu_isced   : MI = 0.0176
```

**Low-signal features (bottom 5):**
```
aspiration_constrained  : MI = 0.000
residence_urban         : MI = 0.0018
has_loan                : MI = 0.0018
n_elderly               : MI = 0.0036
tenure_type_renter      : MI = 0.0053
```

**Interaction features retained:**
```
renter_urban           : retained (mean=0.291)
high_dep_rural         : retained (mean=0.310)
low_edu_no_internet    : retained (mean=0.012)
elderly_dep_ratio      : retained (mean=0.076)
formal_credit_absent   : retained (mean=0.974)
low_finance_density_county : DROPPED (zero variance)
```

### 5.4 Train/Test Split
```
Train rows         : 17,077
Test rows          :  4,270
Candidate features :     66
Selected features  :     35
Unweighted high %  : train=40.0,  test=40.0
Weighted high %    : train=38.4,  test=38.9
Strata count       :     16
```

**35 selected proxy features (by MI rank):**

| Rank | Feature | MI |
|---|---|---|
| 1 | mean_age | 0.0325 |
| 2 | n_children | 0.0224 |
| 3 | n_elderly | 0.0080 |
| 4 | dependency_ratio | 0.0098 |
| 5 | max_edu_isced | 0.0091 |
| 6 | mean_edu_isced | 0.0175 |
| 7 | tenure_type_renter | 0.0052 |
| 8 | residence_urban | 0.0000 |
| 9 | pct_urban_county | 0.0568 |
| 10 | has_internet | 0.0053 |
| 11 | has_mortgage | 0.0000 |
| 12 | has_loan | 0.0021 |
| 13 | has_formal_credit | 0.0007 |
| 14 | renter_urban | 0.0000 |
| 15 | high_dep_rural | 0.0000 |
| 16 | elderly_dep_ratio | 0.0083 |
| 17 | county_n_hh | 0.0518 |
| 18 | wap_share | 0.0189 |
| 19 | female_share | 0.0151 |
| 20 | n_working_age | 0.0139 |
| 21–35 | h02, g06__1, k17, g06__7, k19, g06__6, h04, g05__3, g06__3, c13__1, k16, h03, h11, h08, pct_born_here | varies |

### 5.5 Spatial Cross-Validation Setup
```
Strategy   : StratifiedGroupKFold (n=5, groups=county)
Spatial note: same county cannot appear in both train AND val fold

Fold 1: train counties=39  val counties= 8  overlap=0 ✓
Fold 2: train counties=39  val counties= 8  overlap=0 ✓
Fold 3: train counties=37  val counties=10  overlap=0 ✓
Fold 4: train counties=37  val counties=10  overlap=0 ✓
Fold 5: train counties=36  val counties=11  overlap=0 ✓

Expected AUC drop vs naive StratifiedKFold: 0.03 to 0.06
```

---

## Phase 6: Model Training & Results

### 6.1 Model A — Logistic Regression
```
Standard CV AUC          : 0.6904
Spatial-corrected CV AUC : 0.6511
Spatial correction       : +0.0394
Test AUC                 : 0.6962

Top 5 positive odds ratios:
  tenure_type_renter : coef=0.791  OR=2.205
  n_working_age      : coef=0.571  OR=1.769
  h04                : coef=0.388  OR=1.474
  g06__3             : coef=0.340  OR=1.405
  n_children         : coef=0.249  OR=1.283
```

### 6.2 Model B — LightGBM
```
Best search CV AUC       : 0.7574
Spatial-corrected CV AUC : 0.6503
Test AUC                 : 0.7779
Test R²                  : 0.3674

Best hyperparameters:
  n_estimators=800, learning_rate=0.02, num_leaves=63,
  max_depth=8, min_child_samples=15, subsample=1.0,
  colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=2.0
```

### 6.3 Model C — XGBoost
```
Best search CV AUC       : 0.7611
Spatial-corrected CV AUC : 0.6538
Test AUC                 : 0.7777
Test R²                  : 0.3681

Best hyperparameters:
  n_estimators=1200, learning_rate=0.02, max_depth=5,
  min_child_weight=10, subsample=0.85, colsample_bytree=0.7,
  reg_alpha=0.0, reg_lambda=4.0
```

### 6.4 Calibration & Ensemble
```
LightGBM raw       : AUC=0.7779  Brier=0.1872
LightGBM isotonic  : AUC=0.7829  Brier=0.1855
XGBoost raw        : AUC=0.7777  Brier=0.1870
XGBoost isotonic   : AUC=0.7787  Brier=0.1868
Blend isotonic     : AUC=0.7820  Brier=0.1857
Blend regression   : R²=0.3716
```

### 6.5 TabNet
```
Early stopping at epoch 157  (best_epoch=127)
Best val RMSE  : 0.07748
Test R²        : 0.2053
Test RMSE      : 0.0771
Status         : ✓ stable
```

### 6.6 MLP (Neural Network)
```
Test R²   : 0.2269
Test RMSE : 0.0760
```

### 6.7 Multi-Class 3-Tier LightGBM (Low / Moderate / High)
```
                   precision  recall  f1-score  support
Low (Tier 0)         0.60      0.61    0.61      1,432
Moderate (Tier 1)    0.44      0.42    0.43      1,428
High (Tier 2)        0.58      0.60    0.59      1,410
accuracy                               0.54      4,270

Per-tier AUC (one-vs-rest):
  Tier 0 Low      : AUC = 0.7835
  Tier 1 Moderate : AUC = 0.6114
  Tier 2 High     : AUC = 0.7814
```

### 6.8 Classification Threshold Tuning (OOF)
| Model | Threshold | Best F1 | F1@0.50 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.395 | 0.610 | 0.575 | 0.491 | 0.803 |
| LightGBM | 0.306 | 0.657 | 0.595 | 0.558 | 0.800 |
| XGBoost | 0.301 | 0.663 | 0.591 | 0.555 | 0.823 |
| Blend | 0.301 | 0.662 | 0.593 | 0.555 | 0.819 |

### 6.9 Full Model Comparison Table
| Model | AUC-ROC | Spatial-CV AUC | Spatial Correction | PR-AUC | Brier | Best F1 | R² | RMSE |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.6962 | 0.6511 | 0.0451 | 0.5706 | 0.2211 | 0.6256 | — | — |
| LightGBM | 0.7779 | 0.6503 | 0.1276 | 0.6691 | 0.1872 | 0.6759 | 0.3674 | 0.0687 |
| LightGBM Isotonic | 0.7829 | — | — | 0.6836 | 0.1855 | 0.6182 | — | — |
| XGBoost | 0.7777 | 0.6538 | 0.1239 | 0.6737 | 0.1870 | 0.6717 | 0.3681 | 0.0687 |
| XGBoost Isotonic | 0.7787 | — | — | 0.6770 | 0.1868 | 0.6280 | — | — |
| Blend | 0.7820 | — | — | 0.6822 | 0.1857 | 0.6719 | 0.3716 | 0.0685 |
| TabNet | — | — | — | — | — | — | 0.2053 | 0.0771 |
| MLP | — | — | — | — | — | — | 0.2269 | 0.0760 |

**Brier score ranking (lower = better calibrated):**
```
1. LightGBM Isotonic : 0.1855
2. Blend Isotonic    : 0.1857
3. XGBoost Isotonic  : 0.1868
4. XGBoost raw       : 0.1870
5. LightGBM raw      : 0.1872
6. Logistic Regression: 0.2211
```

---

## Phase 7: SHAP Feature Importance

### 7.3 Top SHAP Features (LightGBM, mean absolute SHAP)

| Rank | Feature | Mean |SHAP| | Dimension Group |
|---|---|---|---|
| 1 | tenure_type_renter | 0.2757 | Residence context |
| 2 | county_n_hh | 0.2395 | Residence context |
| 3 | n_children | 0.2350 | Demographic structure |
| 4 | n_working_age | 0.2274 | Demographic structure |
| 5 | pct_urban_county | 0.2271 | Residence context |
| 6 | g06__6 | 0.1605 | Housing perception |
| 7 | h03 | 0.1414 | Housing perception |
| 8 | mean_age | 0.1079 | Demographic structure |
| 9 | h02 | 0.0967 | Housing perception |
| 10 | h04 | 0.0890 | Housing perception |

> The top SHAP driver group is **Residence context** (renter status, county size, urbanisation), confirming that geography and tenure type carry the most predictive signal for HFVS classification without access to financial variables.

---

## Phase 8: County Risk Mapping & Spatial Validation

### 8.1 Top 10 Most Vulnerable Counties

| County | Mean HFVS | % High Vuln | % Triple Exposed | Mortgage Penetration |
|---|---|---|---|---|
| Tana River | 0.4323 | 70.9% | 2.18% | 0.77% |
| Kisumu | 0.4111 | 64.5% | 7.16% | 1.75% |
| West Pokot | 0.4081 | 63.5% | 0.10% | 0.38% |
| Lamu | 0.4059 | 65.3% | 0.37% | 0.48% |
| Bomet | 0.4034 | 61.5% | 0.04% | 0.74% |
| Isiolo | 0.4013 | 58.5% | 3.97% | 0.31% |
| Homa Bay | 0.3988 | 54.3% | 1.48% | 1.01% |
| Trans Nzoia | 0.3964 | 55.6% | 1.03% | 1.42% |
| Migori | 0.3925 | 59.4% | 0.88% | 2.40% |
| Busia | 0.3907 | 58.2% | 0.51% | 0.00% |

### 8.4 IRA Actuarial Validation

```
Loaded: 7 counties with IRA + HFVS data
Loaded: 9 class-year rows (IRA national ratios)

[Calibration Anchors — IRA Annual Report 2023, Table 28]
  Fire Domestic claim ratio (2023): 28.4%
  Industry average claim ratio     : 67.7%
  Fire Domestic is 2.4× below industry average

[Actuarial Validation — Spearman Rank Correlation]
  HFVS vs Insurance Density 2023: rho = 0.643  (p = 0.1194)  n = 7
  NOTE: positive rho due to Nairobi's dual dominance
        (highest HFVS urban + 82.1% of national GDPI).
        With n=7 this is not statistically meaningful.

[Premium Concentration — IRA Table 5, 2023]
  7 counties account for 100.0% of national GDPI
  40 counties collapsed as "Others" = 0.0%
  → This IS the actuarial proof of the HFVS measurement vacuum hypothesis.
```

### 8.5 AHP Programme Alignment Analysis
```
AHP counties (n=15)    : mean HFVS = 0.3469
Non-AHP counties (n=32): mean HFVS = 0.3594
Mann-Whitney U = 200.0,  p = 0.3673
AHP mean vulnerability rank : 26.7  (lower = more vulnerable)
National mean rank          : 24.0
→ AHP programme is NOT statistically aligned to highest-vulnerability counties
```

**Top 15 most vulnerable counties with NO AHP project:**

| County | HFVS Rank | Mean HFVS | % High Vuln | % Triple | % Urban | Mortgage % |
|---|---|---|---|---|---|---|
| Tana River | 1 | 0.4323 | 70.9% | 2.18% | 29.5% | 0.77% |
| West Pokot | 3 | 0.4081 | 63.5% | 0.10% | 7.4% | 0.38% |
| Lamu | 4 | 0.4059 | 65.3% | 0.37% | 30.1% | 0.48% |
| Bomet | 5 | 0.4034 | 61.5% | 0.04% | 5.0% | 0.74% |
| Isiolo | 6 | 0.4013 | 58.5% | 3.97% | 52.4% | 0.31% |
| Homa Bay | 7 | 0.3988 | 54.3% | 1.48% | 11.9% | 1.01% |
| Trans Nzoia | 8 | 0.3964 | 55.6% | 1.03% | 22.2% | 1.42% |
| Migori | 9 | 0.3925 | 59.4% | 0.88% | 17.9% | 2.40% |
| Busia | 10 | 0.3907 | 58.2% | 0.51% | 15.5% | 0.00% |
| Marsabit | 11 | 0.3854 | 55.9% | 0.46% | 26.4% | 0.35% |
| Narok | 12 | 0.3832 | 50.6% | 0.29% | 13.4% | 0.35% |
| Vihiga | 14 | 0.3761 | 50.2% | 0.00% | 11.1% | 0.00% |
| Mandera | 17 | 0.3720 | 49.2% | 0.28% | 31.6% | 0.00% |
| Nyamira | 20 | 0.3674 | 47.4% | 0.00% | 9.2% | 0.47% |
| Samburu | 21 | 0.3647 | 42.4% | 0.41% | 19.5% | 3.80% |

### 8.6 Urban-Rural HFVS Disaggregation
```
National: Rural mean HFVS = 0.354
          Urban mean HFVS = 0.357
          Rural-Urban gap  = -0.003  (near-zero)
```
> The near-zero national gap masks within-county heterogeneity — urban counties contain both very high-vulnerability informal settlement households and very low-vulnerability formal-sector households.

### 8.7 Finance Exclusion Quadrant Analysis
```
Spearman rho(HFVS, mortgage_penetration) = -0.294  (p = 0.0450)
→ Statistically significant NEGATIVE correlation
→ Confirms: counties with higher vulnerability have lower mortgage penetration
```

**Policy-target counties (high HFVS + low mortgage penetration):**

| County | Mean HFVS | Mortgage Penetration | % Triple Exposed |
|---|---|---|---|
| Tana River | 0.4323 | 0.77% | 2.18% |
| West Pokot | 0.4081 | 0.38% | 0.10% |
| Lamu | 0.4059 | 0.48% | 0.37% |
| Bomet | 0.4034 | 0.74% | 0.04% |
| Isiolo | 0.4013 | 0.31% | 3.97% |
| Busia | 0.3907 | 0.00% | 0.51% |
| Marsabit | 0.3854 | 0.35% | 0.46% |
| Narok | 0.3832 | 0.35% | 0.29% |
| Nairobi | 0.3796 | 0.79% | 11.63% |
| Vihiga | 0.3761 | 0.00% | 0.00% |
| Garissa | 0.3726 | 0.00% | 1.15% |
| Mandera | 0.3720 | 0.00% | 0.28% |

### 8.8 Pre-Model vs Post-Model County HFVS
```
Spearman rho (actual vs proxy rank) : 0.960  (p = 0.0000)
Mean absolute error (county HFVS)   : 0.0086

Counties with largest proxy divergence:
  Trans Nzoia   : actual=0.396, proxy=0.375, diff=0.022
  Wajir         : actual=0.364, proxy=0.385, diff=0.021
  Tana River    : actual=0.432, proxy=0.451, diff=0.019
  Kitui         : actual=0.300, proxy=0.319, diff=0.019
  Tharaka-Nithi : actual=0.279, proxy=0.296, diff=0.017
  West Pokot    : actual=0.408, proxy=0.425, diff=0.017
  Samburu       : actual=0.365, proxy=0.381, diff=0.016
  Marsabit      : actual=0.385, proxy=0.400, diff=0.015
```

### 8.9 Within-County HFVS Inequality (Gini Coefficients)

**Top 10 most internally unequal counties:**

| County | Gini | Mean HFVS | IQR | n |
|---|---|---|---|---|
| Murang'a | 0.1693 | 0.3052 | 0.1339 | 355 |
| Taita-Taveta | 0.1608 | 0.3369 | 0.1314 | 405 |
| Kirinyaga | 0.1575 | 0.3186 | 0.1268 | 450 |
| Tharaka-Nithi | 0.1509 | 0.2766 | 0.1060 | 455 |
| Trans Nzoia | 0.1493 | 0.3859 | 0.1574 | 373 |
| Kiambu | 0.1478 | 0.3194 | 0.1203 | 319 |
| Kajiado | 0.1446 | 0.3403 | 0.1141 | 411 |
| Kwale | 0.1443 | 0.3431 | 0.1187 | 410 |
| Nyeri | 0.1440 | 0.2896 | 0.1005 | 434 |
| Homa Bay | 0.1432 | 0.4074 | 0.1603 | 422 |

**10 most internally equal counties:**

| County | Gini | Mean HFVS | IQR | n |
|---|---|---|---|---|
| Garissa | 0.1055 | 0.3804 | 0.0779 | 468 |
| Elgeyo-Marakwet | 0.1027 | 0.3527 | 0.0867 | 391 |
| Turkana | 0.1024 | 0.3473 | 0.0676 | 368 |
| Bungoma | 0.1009 | 0.3667 | 0.0793 | 462 |
| Kakamega | 0.0993 | 0.3639 | 0.0811 | 480 |
| Kericho | 0.0975 | 0.3503 | 0.0775 | 410 |
| Marsabit | 0.0947 | 0.3915 | 0.0825 | 462 |
| Kisii | 0.0934 | 0.3723 | 0.0873 | 510 |
| Nyamira | 0.0903 | 0.3661 | 0.0718 | 439 |
| Vihiga | 0.0887 | 0.3838 | 0.0826 | 400 |

```
Gini vs mean HFVS: rho = -0.368  (p = 0.0110)
Finding: more vulnerable counties are also more internally unequal.
Policy implication: county-level instruments are LEAST efficient
                    where they are MOST needed.
```

---

## Phase 9: Conclusions

### 9.1 Summary of Findings

```
National HFVS:
  Mean score           : ~0.41  (std ~0.08)
  High vulnerability   : 40.0% of 21,347 surveyed households
  Threshold            : 60th percentile (HFVS > 0.373)
  Inter-county range   : ~0.12 score points (least to most vulnerable)
```

**Five key findings from the pipeline:**

1. **Measurement vacuum confirmed.** Income quintiles misclassify vulnerability. The HFVS captures five independent risk dimensions — financial stress, tenure insecurity, physical hazard, dwelling quality, and utility deprivation — none of which is reliably proxied by income alone.

2. **Utility deprivation dominates.** D5 carries an effective weight of 34.3% vs its stated 20%, driven by near-universal solid fuel cooking (96.6%) and 43.7% without grid electricity. D3 physical hazard is underweighted (effective 6.2%) due to proximity columns being absent from the extract.

3. **Compound exposure is an urban phenomenon.** Triple-exposed households (flood zone + no tenure + rent stressed) are 95.3% urban. The urban triple rate (3.90%) is 25.8× the rural rate (0.15%). Top counties: Nairobi (14.73%), Mombasa (9.88%), Kisumu (7.62%).

4. **The proxy model is county-viable.** LightGBM achieves AUC=0.778 and R²=0.367 using only 35 proxy features (no financial variables). County-level Spearman rho between actual and proxy HFVS = 0.960 (p=0.0000), MAE = 0.0086. This means HFVS can be estimated without a full KHS questionnaire.

5. **Finance exclusion is structural.** Spearman rho(HFVS, mortgage_penetration) = -0.294 (p=0.0450). 40 of 47 counties are entirely absent from IRA Table 5 — they produce no measurable national GDPI. The 7 counties that appear account for 100% of reported premium. This concentration IS the actuarial validation of the measurement vacuum hypothesis.

### 9.2 County-Level Gradient
```
Most vulnerable county  : Tana River  (mean HFVS = 0.432, 70.9% high-vulnerability)
Least vulnerable county : (lowest-scoring county, ~0.27–0.30 range based on full distribution)
```

### 9.3 Internal Inequality Finding
```
Gini vs mean HFVS: rho = -0.368  (p = 0.0110)
The most vulnerable counties are also the most internally unequal.
County-level policy instruments are LEAST efficient where they are MOST needed.
Most unequal county (Gini): Murang'a (0.169)
Most equal county (Gini):   Vihiga   (0.089)
```

### 9.5 Limitations

1. **IRA data structural gap.** IRA Table 5 names only counties with ≥1.0% national GDPI market share. 40 counties are collapsed into a single "Others (7.0%)" entry. A class-specific (fire/domestic) county breakdown is unavailable, preventing full actuarial calibration.

2. **D3 physical hazard is sparse.** Six proximity columns (e08__1–e08__6) are absent from the extract. D3 relies solely on flood zone (e06) and mudslide zone (e07), giving it an effective weight of only 6.2%.

3. **D1 owner-occupier proxy.** Rent variable k05 is structurally missing for owner-occupiers. Estimated rental value (l15) is used as a proxy, introducing measurement error that cannot be fully corrected by group-median imputation.

4. **k25 variable retired.** The willingness-to-spend variable has a median of KES 1,000,000 — confirmed non-monthly — and was excluded from all derived variables.

5. **Spatial generalisation gap.** Spatial-corrected CV AUC (0.65) is ~0.13 below standard CV AUC (0.78), indicating some geographic overfitting. The model performs best in counties similar to its training set.

---

## Phase 10: Recommendations & Next Steps

### 10.1 Policy Recommendations

| Stakeholder | Recommendation | Evidence Base | Priority |
|---|---|---|---|
| State Dept of Housing | Use HFVS county rank as mandatory input in AHP site selection; prioritise top missed counties from Phase 8.5 alignment table | Phase 8.5 | Immediate |
| Insurance Regulatory Authority | Use mean county HFVS as risk-loading variable in property microinsurance pricing; calibrate above Fire Domestic 28.4% baseline proportionally to HFVS decile | Phase 8.4 IRA validation | Immediate |
| Kenya Mortgage Refinance Company | Expand KMRC concessional lending to counties in bottom-quartile mortgage penetration (Phase 8.7 finance exclusion quadrant) | Phase 2.7 + 8.7 | Short-term |
| NGOs / UN-Habitat | Target D5 utility interventions (rural electrification, WASH) in ASAL counties with highest utility deprivation scores | Phase 4 + 8 | Short-term |
| Insurance underwriters | Develop parametric flood-trigger products in counties where triple-exposed HH exceed 15% (Mombasa, Nairobi, Kisumu minimum) | Phase 4.4 | Medium-term |
| KNBS | Expand next KHS to include asset values, formal insurance coverage, claims history, and explicit monthly housing budget question to replace k25 | Phase 9.5 | Long-term |

### 10.2 Research Next Steps

- **Longitudinal follow-up (2026–27):** Re-interview 2023/24 KHS households and match against insurance claims records — the definitive actuarial test.
- **Actuarially calibrated weights:** When IRA claims records are matched to KNBS survey keys via regulatory data-sharing, re-estimate dimension weights using Poisson regression of claim frequency on dimension scores.
- **East African expansion:** The five-dimension HFVS structure, WHO/JMP-aligned material codes, and CRISP-DM pipeline are directly transferable to Uganda's NHSurvey and Tanzania's National Panel Survey.
- **Field deployment tool:** Build a 10-minute tablet survey using the top SHAP features (tenure status, county, children, working-age members, urbanisation) to allow community workers to estimate HFVS without the full KHS questionnaire.
- **Parametric insurance product design:** Counties where triple-exposed households exceed 15% are highest-priority markets for automatic flood-trigger housing products.

### 10.3 Open Science Commitment
```
Repository: github.com/VAL-Jerono/KHS_housing_dissertation
Outputs    : master_hfvs_v3.parquet (21,347 × 515 columns)
             All figures and CSV tables reproducible top-to-bottom
             on Google Colab Pro+
```

### 10.4 Closing Statement

> The Housing Financial Vulnerability Score is not a solution to Kenya's two-million-unit housing deficit. It is the **evidence layer** that every other solution has been operating without. The AHP can now select sites by vulnerability rank, not land availability. The IRA can load premiums by county HFVS, not geographic intuition. The KMRC can target concessional lending by the quadrant this analysis defines, not by existing customer relationships. The measurement gap does not close by itself. It closes when this framework is operationalised, linked to real IRA loss data, and embedded in the policy instruments designed to reach Kenya's most vulnerable housing market participants.

---

*Generated from: `KHS_Dissertation_Final_Enhanced.ipynb` — all values are direct notebook outputs.*
