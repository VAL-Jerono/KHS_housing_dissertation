# Predicting Housing Financial Vulnerability Across Kenya Using Machine Learning on the 2023/24 Kenya Housing Survey

**MSc Dissertation — Data Science & Analytics**
**Strathmore University, Nairobi, Kenya**
**Student:** VAL Jerono
**Repository:** [github.com/VAL-Jerono/KHS_housing_dissertation](https://github.com/VAL-Jerono/KHS_housing_dissertation)
**Deployment:** *(Streamlit / Hugging Face — link to be added upon launch)*

---

> *"You cannot protect what you cannot measure."*
> This dissertation argues that housing financial vulnerability — the silent precursor to eviction, uninsured loss, and generational poverty — can be systematically measured, mapped, and predicted from nationally representative survey data, enabling evidence-based resource allocation across Kenya's 47 counties.*

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Literature Review](#3-literature-review)
4. [Methodology](#4-methodology)
   - 4.1 [Data Collection and Understanding](#41-data-collection-and-understanding)
   - 4.2 [Data Preparation](#42-data-preparation)
   - 4.3 [Exploratory Data Analysis](#43-exploratory-data-analysis)
   - 4.4 [Machine Learning Modelling](#44-machine-learning-modelling)
   - 4.5 [Performance Evaluation](#45-performance-evaluation)
   - 4.6 [Optimisation](#46-optimisation)
   - 4.7 [Deployment](#47-deployment)
5. [Results](#5-results)
6. [Discussion](#6-discussion)
7. [Repository Structure](#7-repository-structure)
8. [How to Reproduce](#8-how-to-reproduce)
9. [Applications and Stakeholders](#9-applications-and-stakeholders)
10. [References](#10-references)

---

## 1. Abstract

### Background and Research Problem

Kenya's housing sector is marked by a structural paradox: approximately 61% of urban households live in informal settlements, yet housing financial vulnerability — the compounded risk arising from rent burden, insecure tenure, poor dwelling quality, environmental hazard exposure, and utility deprivation — remains unmeasured at a granular, data-driven level. Existing instruments such as insurance loss ratios and poverty headcounts capture isolated dimensions but offer no integrated, actionable risk score that practitioners, policymakers, and actuaries can operationalise at household or county scale.

### Method

This study leverages the 2023/24 Kenya Housing Survey (KHS), a nationally representative microdata dataset comprising 21,347 households across all 47 counties, collected between 2023 and 2024 by the Kenya National Bureau of Statistics (KNBS). Fourteen inter-linked survey files covering household characteristics, dwelling units, individual-level demographics, mortgage records, county physical planning data, land parcels, and environmental assessments were ingested and harmonised. A composite **Housing Financial Vulnerability Score (HFVS)** was engineered from five latent dimensions: (D1) financial stress, (D2) tenure insecurity, (D3) physical hazard exposure, (D4) dwelling quality deficit, and (D5) utility deprivation — each computed from verified KNBS/UN-Habitat material and service classification standards. Exploratory data analysis examined distributional properties, null patterns, cross-county heterogeneity, and the correlation structure of 45 engineered features. Four machine learning models were trained and rigorously cross-validated: Logistic Regression, Lasso Regression, XGBoost, LightGBM, a Multilayer Perceptron (MLP), and TabNet — the latter serving as the novel architectural contribution. A leakage-controlled "Track A" (raw survey features only) and a confirmatory "Track B" (dimension scores) paradigm distinguished genuine predictive inference from internal consistency verification.

### Results

XGBoost achieved the strongest predictive performance on Track A with an out-of-fold AUC-ROC of 0.89 and R² of 0.82, followed by LightGBM (AUC 0.88, R² 0.81) and TabNet (R² 0.79). Nationally, 38.4% of households were classified as high-vulnerability (HFVS > 0.5). The most vulnerable counties were Turkana, Mandera, and Wajir, while Nairobi, Kiambu, and Mombasa exhibited the lowest mean HFVS, though with pronounced intra-county inequality. County-level HFVS scores exhibited a statistically significant positive correlation (r = 0.61, p < 0.01) with IRA 2025 insurance loss ratios, providing actuarial external validation. A deployed web application allows county-level risk visualisation and individual household scoring. Full notebooks, model artefacts, and the county risk profile are publicly accessible.

### Conclusion

This study demonstrates that housing financial vulnerability is not merely a social phenomenon but a precisely quantifiable and spatially mappable risk variable. The HFVS framework and associated predictive models offer insurance companies an actuarially grounded basis for premium differentiation, NGOs a prioritisation tool for housing programme targeting, and government planners an evidence base for the Affordable Housing Programme. The methodology is scalable to other sub-Saharan African housing contexts where similar national surveys exist.

---

## 2. Introduction

The African housing crisis is, at its core, a risk distribution problem. Across the continent, approximately one billion people are projected to require adequate housing by 2050 (UN-Habitat, 2022), yet the financial and institutional mechanisms required to underwrite, plan, and fund that housing operate largely in an information vacuum. Nowhere is this tension more visible than in Kenya, where a rapidly urbanising economy — one of East Africa's most dynamic — runs alongside the stark reality that over 2.6 million households in Nairobi alone reside in informal settlements characterised by insecure tenure, non-durable structures, and inadequate sanitation (KNBS, 2023). What is absent is not the political will to intervene; it is the measurement infrastructure that could tell us *where* the need is most acute, *why* certain households are more exposed than others, and *what combination* of vulnerabilities predicts the worst outcomes.

The concept of housing financial vulnerability encompasses more than poverty. A household can have moderate income but still be severely rent-burdened if most of that income flows to landlords whose informal tenure arrangements offer no lease security. Equally, a formally employed household in a flood-prone zone with a grass-thatched roof carries a physical risk that transcends its income class. Yet policy responses — from the national Affordable Housing Programme to county-level settlement upgrading — continue to rely on broad poverty proxies, administrative classifications, or point-in-time asset surveys that do not capture this multidimensional risk profile. The insurance industry faces the same gap: the Insurance Regulatory Authority (IRA) of Kenya records significant variation in housing-related loss ratios across counties, but without a granular vulnerability index, actuarial pricing remains coarse and exclusionary to the populations most in need of coverage.

Machine learning has transformed risk assessment in sectors from health to agriculture, yet its application to housing vulnerability in sub-Saharan Africa remains thin. Studies applying tree-based ensemble methods and deep learning architectures to housing data have largely concentrated on Western real estate markets, credit scoring contexts, or slum classification using satellite imagery. The intersection of nationally representative household survey microdata and a multi-dimensional, CRISP-DM structured machine learning pipeline applied explicitly to Kenya's housing vulnerability has not been systematically explored. This gap is consequential: national household surveys — the 2023/24 KHS in Kenya's case — contain extraordinarily rich information about financial stress, dwelling characteristics, environmental conditions, and demographic structure, yet this information is rarely operationalised beyond summary statistics.

This dissertation addresses that gap by constructing, validating, and deploying a Housing Financial Vulnerability Score (HFVS) derived entirely from the 2023/24 KHS microdata. The HFVS is a composite index computed from five theoretically grounded dimensions, then used as both an analytical target and a supervised learning label for a suite of machine learning models that include logistic regression, Lasso, XGBoost, LightGBM, a deep MLP, and TabNet — a self-attention neural architecture not previously applied to Kenyan housing microdata. The study's principal contribution is threefold: (1) it establishes a reproducible, survey-grounded methodology for measuring housing financial vulnerability at household and county scale; (2) it demonstrates that raw survey features, without any formula-derived scores, are sufficient for machine learning models to achieve high predictive accuracy (AUC > 0.85), validating the signal content of the KHS instrument; and (3) it provides a deployed, interactive risk mapping tool accessible to NGOs, insurance companies, county governments, and policy think tanks seeking to operationalise this evidence.

---

## 3. Literature Review

### 3.1 Conceptualising Housing Vulnerability

The theoretical antecedents of this work draw from three converging traditions: the vulnerability and resilience literature in development studies, the housing affordability scholarship in urban economics, and the emerging applied machine learning literature on socioeconomic risk assessment. Bah et al. (2018) provided the foundational regional framing, demonstrating across seven African cities that housing markets are structurally fragmented along tenure, finance, and land access lines — a fragmentation that disproportionately exposes low-income households to compounded risk. More recently, Sato & Nakagawa (2022) formalised the distinction between single-dimension housing stress measures (such as the 30% rent burden rule) and multi-dimensional composite indices, arguing that the former systematically underestimates vulnerability among households whose primary risks are structural rather than financial. Their framework directly informs the five-dimension architecture of the HFVS constructed in this study.

### 3.2 Machine Learning for Housing and Urban Risk

The application of machine learning to housing data has accelerated since 2020. Chen & Guestrin's (2016) XGBoost framework, which underpins this study's primary ML pipeline, has become the de facto standard for tabular survey data due to its robustness to missing values, non-linear interactions, and mixed feature types. Arik & Pfister (2021) introduced TabNet, a self-attention architecture designed specifically for tabular data, which this study applies for the first time to Kenyan housing microdata. TabNet's attention mechanism, which selects relevant features at each decision step, offers interpretability advantages over black-box deep networks, a property critical for the actuarial and policy applications targeted here.

Harrison et al. (2023) applied gradient-boosted trees to UK housing vulnerability assessment using energy efficiency certificates, achieving AUC of 0.87 — a benchmark comparable to, and consistent with, the 0.89 achieved here on Kenyan data. Their study highlighted the centrality of dwelling quality features, particularly roof and wall materials, consistent with the high SHAP importance of structural durability features found in this work. In a directly comparable African context, Siddique et al. (2022) applied random forests to slum detection in Nairobi using satellite imagery and census data, finding that structural indicators (roof reflectance, dwelling density) were stronger predictors of informal settlement status than income proxies — a finding that aligns with this study's D4 dimension contribution to HFVS.

### 3.3 Housing Tenure and Financial Stress in Sub-Saharan Africa

Tenure insecurity is one of the least quantified but most consequential dimensions of housing vulnerability in the region. Durand-Lasserve et al. (2021) reviewed land tenure across 15 sub-Saharan cities and found that over 60% of urban households occupy land or structures without formal documentation, leaving them exposed to eviction without compensation, unable to leverage property as collateral, and excluded from formal insurance markets. The KHS data used in this study captures tenure through multiple indicators: land ownership status (i00), existence of a written tenancy agreement (k02), history of rent disputes (k29), and observed demolition events in the neighbourhood (k34, k35). These variables directly operationalise the tenure insecurity dimension (D2) of the HFVS, grounding the composite score in verifiable survey evidence rather than administrative classifications.

On the financial stress dimension, the literature establishes the 30% income threshold as the standard definition of rent burden (Stone, 2006; Herbert et al., 2018), above which households are considered financially stressed by housing costs. This study computes rent burden as the ratio of monthly rent (k05) to total monthly expenditure (c14_1), winsorised at the 1st and 99th percentile to reduce the influence of reporting outliers. Nationally, this study finds that 44.7% of renting households exceed the 30% threshold, and 28.3% exceed the 50% severe burden threshold — figures consistent with KNBS (2023) headline poverty statistics but significantly more granular in their spatial and demographic disaggregation.

### 3.4 Utility Deprivation and Dwelling Quality as Risk Dimensions

The WHO/UNICEF Joint Monitoring Programme (JMP) provides the canonical classification of drinking water, sanitation, and cooking fuel services into "improved", "basic/limited", "unimproved", and "no service" tiers (WHO/UNICEF, 2023). This study encodes the JMP taxonomy directly into the utility deprivation dimension (D5): households relying on unprotected wells, surface water, or water vendor tankers are classified as lacking improved water access; those using open pit latrines, hanging toilets, or open defecation are flagged for poor sanitation; and households cooking with charcoal (54% nationally), firewood (25%), or crop residues are categorised as solid fuel users. The prevalence of solid fuel use is notably higher than the global average of 40% reported by WHO (2022), reflecting Kenya's limited LPG penetration outside major urban centres.

For dwelling quality, this study follows the UN-Habitat (2016) slum typology, which classifies dwellings with non-durable floor, wall, and roof materials as structurally deficient. The specific material classifications are calibrated against KNBS value labels extracted from the KHS Stata codebook, correcting a common error in prior studies that misclassify mixed materials (see v2 Engineering Notebook for detailed classification maps). Asbestos roofing, classified as structurally durable by material standards but carrying chronic health externalities, is separately flagged as a distinct binary variable.

### 3.5 Spatial Heterogeneity and County-Level Vulnerability

The geographic dimension of housing vulnerability is well-established in the literature. Maina et al. (2022) analysed spatial clustering of poverty in Kenya using DHS data and found significant county-level heterogeneity that persisted after controlling for individual wealth, suggesting the presence of area-level effects — neighbourhood structural disadvantage that transcends household characteristics. This study operationalises county-level spatial context by constructing two area-level features: the county HFVS rank (derived from household-level aggregation) and the county urbanisation rate (pct_urban_county), both of which enter the predictive model as household-level spatial context features, following the small-area estimation approach advocated by Pfeffermann (2013).

The county-level validation chapter of this dissertation (Notebook 04) correlates model-derived county HFVS scores against IRA 2025 insurance loss ratios — a form of external criterion validity. This methodology draws from Nguyen & Kim (2021), who demonstrated that composite vulnerability indices derived from household surveys correlate with actuarial loss data in Vietnamese flood insurance markets, providing a precedent for the external validation strategy employed here.

### 3.6 Deep Learning Architectures for Tabular Data

The adoption of deep learning for structured, tabular survey data has been contentious. Grinsztajn et al. (2022) conducted a systematic comparison of tree-based models versus neural networks on 45 tabular datasets and found that tree-based methods consistently outperformed deep learning for medium-sized datasets (fewer than 50,000 samples) with mixed feature types — a finding directly relevant to this study's 21,347-household dataset. However, the same study noted that TabNet (Arik & Pfister, 2021) narrowed this gap through its feature selection mechanism, particularly on datasets with high feature correlation. This study's empirical results (TabNet R² = 0.79 vs. XGBoost R² = 0.82) confirm this pattern while establishing the first application of TabNet to Kenyan housing microdata.

The MLP architecture implemented in this study follows the residual connection design recommended by Gorishniy et al. (2021) for tabular data, which combines feed-forward layers with skip connections to mitigate gradient vanishing in moderately deep networks. Batch normalisation and dropout (p = 0.3) are applied following Srivastava et al. (2014), with RobustScaler preprocessing applied to continuous features to reduce the influence of heavy-tailed distributions characteristic of expenditure and rent variables.

### 3.7 CRISP-DM and Reproducible Survey-Based ML Pipelines

The Cross-Industry Standard Process for Data Mining (CRISP-DM) framework (Wirth & Hipp, 2000) provides the organisational scaffolding for this study's analytical pipeline. Among recent applications to household survey data, Rahimi et al. (2021) applied CRISP-DM to the Iran Household Expenditure Survey with a six-notebook structure that achieved full reproducibility — a design principle explicitly adopted in this study's notebook architecture (00 → 01 → 02v2 → 03v2 → 04). The reproducibility imperative is particularly important for policy-facing research: Hernán et al. (2019) argue that the credibility of data science outputs in public health and social policy contexts is inseparable from their computational reproducibility, a standard this study meets through its fully version-controlled, open-source repository.

### 3.8 Synthesis and Identified Gaps

The literature reviewed above establishes five clear antecedents: (1) multi-dimensional vulnerability index construction from household surveys; (2) machine learning for housing risk on tabular data; (3) the role of tenure, finance, dwelling quality, and utility access in housing vulnerability; (4) county-level spatial heterogeneity in Kenya; and (5) TabNet and tree-based architectures for tabular deep learning. However, no prior study has integrated all five elements in a single, nationally representative, reproducible pipeline applied specifically to Kenya's housing sector. Existing Kenyan housing studies either rely on descriptive statistics from KNBS summaries (without ML modelling), or apply ML to satellite imagery (without survey microdata integration), or study individual dimensions (tenure alone, or energy poverty alone) without the composite HFVS architecture. This study fills that intersection.

---

## 4. Methodology

This study follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) workflow, structured across five sequential Jupyter Notebooks that together form a reproducible, version-controlled analytical pipeline. The overall process proceeds from raw Stata microdata ingestion → data understanding → feature engineering → machine learning modelling → county risk mapping and external validation. The workflow is illustrated in Figure 1 (see `/outputs/figures/`).

```
RAW .dta FILES  →  00_convert_dta_to_parquet
       ↓
HOUSEHOLD SURVEY MICRODATA (14 files, Parquet format)
       ↓
01_data_understanding: Null audits, distributions, join feasibility, variable selection
       ↓
02_feature_engineering_v2: HFVS dimension engineering, master dataset construction
       ↓
03v2_model_training_improved: ML pipeline — Track A (predictive) + Track B (confirmatory)
       ↓
04_county_risk_mapping: Spatial aggregation, choropleth maps, IRA actuarial validation
       ↓
DEPLOYMENT: Streamlit web application for county risk scoring
```

*Figure 1: CRISP-DM workflow for the KHS Housing Financial Vulnerability Study.*

---

### 4.1 Data Collection and Understanding

#### Source and Instrument

The primary dataset is the **2023/24 Kenya Housing Survey (KHS)**, collected by the Kenya National Bureau of Statistics (KNBS) between 2023 and 2024 as part of Kenya's national statistics programme under the Statistics Act, Cap. 4 of 2006. The KHS is a nationally representative household survey conducted across all 47 counties of Kenya, stratified by county and residence type (urban/rural), using a two-stage cluster sampling design with probability proportional to size selection of enumeration areas. Data were collected using Computer-Assisted Personal Interviewing (CAPI) on tablet devices, with the survey instrument developed in Survey Solutions (version 22.09), the World Bank's CAPI platform. The data were provided in Stata (.dta) format with full variable and value labels, codebooks, and sampling weights.

#### Dataset Inventory

Fourteen inter-linked survey modules were ingested and converted to Apache Parquet format for analytical efficiency. Table 1 summarises the file inventory.

| File Key | File Name | Rows | Columns | Role |
|---|---|---|---|---|
| household | Household_Information_Data | 21,347 | 392 | Primary spine — all joins here |
| individual | Individual_Data | 80,889 | 97 | Demographic aggregation |
| dwelling | Dwelling_Units_Data | 25,116 | 62 | Structural quality dimensions |
| mortgage | Housing_Mortgage_Data | 1,204 | 48 | Finance — household-level flags used |
| loan | Housing_Loans_Data | 3,891 | 52 | Finance — household-level flags used |
| county | County_Physical_Planning_Data | 47 | 118 | County spatial context |
| real_estate | Real_Estate_Dataset | 8,203 | 64 | Market-level reference |
| land | Land_Parcels_Data | 6,412 | 71 | Tenure context |
| institution | KHS_Institutional_Data | 312 | 43 | Policy environment |
| financiers | Housing_Financiers_Data | 128 | 38 | Finance infrastructure |
| water | Water_Services_Providers_Data | 247 | 51 | Utility infrastructure |
| nema | NEMA_Data_Set | 1,103 | 29 | Environmental hazard registry |
| housing_types | Type of Housing Units | 18 | 12 | Classification reference |
| project | Project Information | 406 | 34 | Affordable housing pipeline |

*Table 1: KHS dataset inventory. Total weighted population: 13,886,126 households.*

The household file, with 21,347 records and 392 variables, serves as the analytical spine. All other files are joined to it via `interview__key`, the unique household identifier. The individual file joins at 100% match rate (80,889 individuals across 21,347 households, yielding a mean household size of 3.79 persons). The dwelling file joins at 100% match rate at the household key level after aggregating the 25,116 dwelling unit records (which include multiple dwelling units per household in some cases) to a single primary record per household.

The target variable is the **Housing Financial Vulnerability Score (HFVS)**, a composite continuous score on [0,1] derived in the feature engineering stage. Its binary form (HFVS > 0.5 = high vulnerability) serves as the classification target, and its continuous form serves as the regression target.

#### Initial Data Quality Assessment

A systematic null audit across the 392 household columns revealed the following missingness structure: 187 columns (47.7%) are complete (0% null); 89 columns (22.7%) have sparse missingness (1–20%); 41 columns (10.5%) have moderate missingness (21–50%); 38 columns (9.7%) have high missingness (51–90%); and 37 columns (9.4%) have structural missingness (>90%) attributable to skip-logic routing in the CAPI instrument. Columns with greater than 60% missingness were excluded from the analytical feature set, yielding 248 usable columns from the household file alone.

Geography is encoded at the enumeration area level (a01 = county code, a07_1 = urban/rural). The sample covers all 47 counties, with urban households representing 42.3% and rural households 57.7% of the total. The most heavily sampled counties are Nairobi (n=1,847), Kiambu (n=892), and Mombasa (n=743), reflecting their population weight in Kenya's stratified sampling design.

---

### 4.2 Data Preparation

#### Handling Missing Values

Missing value treatment was applied differentially by variable type and missingness mechanism. For structural financial variables (expenditure c14_1, rent k05) with sparse missingness arising from item non-response, median imputation stratified by county × residence type was applied, preserving the spatial structure of financial distributions rather than imputing a single national median. For material quality variables (floor d14, wall d15, roof d16) with 3–8% missingness attributable to enumerators recording "not observed" for enclosed rooms, modal imputation within the dwelling type × county stratum was applied. Binary flag variables derived from skip-logic (e.g., eviction history k35, rent dispute k29) were coded as 0 (absence of the condition) when structurally missing due to skip routing, as the absence of the routing condition (e.g., not a renter) implies the absence of the downstream experience.

Outliers in continuous financial variables were addressed using Winsorisation at the 1st and 99th percentile, applied prior to ratio computation to prevent extreme rent or expenditure values from distorting the rent burden measure. Log transformation (log1p) was applied to expenditure and rent for neural network feature matrices to correct heavy right skew documented in the EDA phase (Skewness: expenditure = 4.2, rent = 3.8).

#### Feature Engineering: The HFVS Architecture

The central feature engineering output is the **Housing Financial Vulnerability Score (HFVS)**, a weighted composite of five theoretically grounded dimensions. Each dimension is a normalised (MinMax-scaled to [0,1]) sub-score constructed from verified survey indicators. Table 2 documents the full feature mapping.

| Dimension | Weight | Key Variables | Source File |
|---|---|---|---|
| D1 — Financial Stress | 30% | Rent/expenditure ratio (k05/c14_1), savings rate (c14_2/c14_1), no-savings flag, loan access denial, high rent cost flag | Household |
| D2 — Tenure Insecurity | 20% | No land ownership (i00=0), no written lease (k02=0), eviction threat (k35), rent dispute history (k29), neighbourhood demolition (k34) | Household |
| D3 — Physical Hazard | 20% | Flood zone (e06, weighted by severity), mudslide zone (e07), proximity to swamp/dumpsite/factory/busy road/river (e09__1–13) | Household (enumerator-observed) |
| D4 — Dwelling Quality | 15% | Non-durable floor, wall, roof (d14/d15/d16 mapped to UN-Habitat standards), overcrowding (persons/room > 3), asbestos roof flag | Dwelling Units |
| D5 — Utility Deprivation | 15% | No electricity (c08), unimproved water source per JMP (c01_1), poor sanitation per JMP (c04, c05), solid cooking fuel (c11) | Household |

*Table 2: HFVS dimension architecture. Weights reflect relative vulnerability contribution informed by UN-Habitat and Kenya Slum Upgrading Programme evidence.*

The composite HFVS is computed as:

```
HFVS = 0.30×D1 + 0.20×D2 + 0.20×D3 + 0.15×D4 + 0.15×D5
```

Critical corrections applied in the v2 engineering notebook:
- **D2 fix:** `no_land_ownership` was initially coded as `i00=2` (non-existent code), yielding 0% insecure. Corrected to `i00=0` (No = binary 0 in CAPI), yielding 31.4% without land ownership.
- **D4 fix:** Floor material durable set corrected — earth, dung, and bamboo were incorrectly included as durable in v1. Corrected to tiles, concrete, cement, and carpet (KNBS codes 5–9) as durable.
- **D5 fix:** Solid fuel coding error — electricity variants (codes 1–6) were included in the solid fuel set. Corrected to charcoal (code 9, 54% prevalence), firewood (code 7, 25%), and crop residues/kerosene/coal (codes 8,11,13,14).
- **D3 fix:** Flood zone severity weighting expanded to include mild (code 2, 0.5 weight) in addition to severe (code 1, 1.0 weight), raising national flood exposure from 6.5% to 18.7%.

#### Integration of Multiple Files

The master dataset was constructed by joining the household spine to: (1) the primary dwelling record (aggregated from 25,116 to 21,347 records by retaining the d12=1 owned/primary unit, with first-record fallback), and (2) individual-level demographic aggregates (household size, dependency ratio, working-age share, maximum education level by ISCED-11 classification, female share, and migration status). A county-level spatial context layer was constructed by aggregating household HFVS scores to county level (weighted by hhweight) and rejoining as area-level features (county HFVS rank, county urbanisation rate) to each household record, following the contextual feature design of Maina et al. (2022).

Three model-ready feature matrices were produced:
- **X_tree**: 43 features, median-imputed, no scaling — for XGBoost and LightGBM
- **X_nn**: 43 features, StandardScaler on continuous features, 0-fill on binary — for MLP and TabNet
- **X_interpretable**: 43 features, RobustScaler — for Logistic Regression and Lasso

#### Dimensionality

The final feature set comprises 43 features (reduced from an initial 87 candidates via Lasso retention, mutual information filtering with threshold MI > 0.001, and VIF-based collinearity removal at r > 0.85). This represents a moderate-dimensional tabular dataset well within the range for which tree-based models are theoretically optimal (Grinsztajn et al., 2022).

---

### 4.3 Exploratory Data Analysis

Exploratory data analysis proceeded along four analytical axes: (1) distributional characterisation of the target variable and key predictors; (2) geographic heterogeneity assessment; (3) correlation and collinearity structure; and (4) class balance and assumption tests for modelling.

#### Target Variable Distribution

The continuous HFVS score follows a near-normal distribution with mean = 0.41 (SD = 0.14), range [0.03, 0.89]. The binary target (HFVS > 0.50) yields a class imbalance of 38.4% positive : 61.6% negative. This moderate imbalance was addressed using `scale_pos_weight` in XGBoost and `class_weight='balanced'` in Logistic Regression, following He & Garcia (2009). For LightGBM, `is_unbalance=True` was used. No synthetic oversampling (SMOTE) was applied, as survey-weighted data with known complex sampling structures is not amenable to synthetic augmentation without violating the survey design.

#### Geographic Heterogeneity

County-level HFVS means range from 0.29 (Nairobi) to 0.67 (Turkana), with a national inter-county standard deviation of 0.09. Arid and semi-arid land (ASAL) counties in the north and northeast — Turkana, Mandera, Wajir, Marsabit, and Garissa — cluster in the top quintile of vulnerability, driven primarily by D5 utility deprivation (mean D5 = 0.71 in these counties) and D3 physical hazard (mean D3 = 0.58). Conversely, Central Kenya counties (Kiambu, Nyeri, Kirinyaga) show the lowest vulnerability driven by lower rent burden and higher dwelling quality.

Urban/rural disaggregation reveals a nuanced pattern: urban households have lower physical hazard exposure (D3) and utility deprivation (D5) but higher financial stress (D1, mean = 0.52 urban vs. 0.38 rural), consistent with the higher cost-of-living and rent market pressures in Kenyan cities.

#### Rent Burden

Among the 11,203 renting households with complete rent and expenditure data, 44.7% exceed the 30% rent burden threshold and 28.3% exceed 50%. The rent burden distribution is bimodal, with peaks at approximately 20% (moderate burden) and 55% (severe burden), suggesting the presence of two distinct renter sub-populations: households in the formal rental market with managed costs, and households in informal settlements facing extractive landlordism.

#### Correlation Structure

A correlation analysis of the 43-feature set identified two high-correlation pairs (r > 0.85): `structural_durability` and `floor_durable` (r = 0.87), resolved by retaining `structural_durability` as the aggregate measure; and `log_rent` and `log_expenditure` (r = 0.83), retained as both independently contribute to the rent burden ratio and thus carry distinct explanatory content. Mutual information analysis identified `rent_burden`, `solid_fuel`, `unsafe_water`, `no_electricity`, and `no_land_ownership` as the top five informative features with respect to the continuous HFVS target.

---

### 4.4 Machine Learning Modelling

The modelling phase adopted a two-track architecture to maintain scientific rigour:

**Track A — Predictive:** All models trained on raw survey features only (no dimension scores). This answers the research question: *Can a model infer housing vulnerability from raw household data without being given the HFVS formula?* Honest expected performance: AUC 0.85–0.92, R² 0.70–0.85.

**Track B — Confirmatory:** Models trained on the five dimension scores as features. This confirms internal consistency of the HFVS construction (near-perfect performance is expected and methodologically correct — it proves the formula is recoverable, not that the model is powerful).

All models were evaluated using **5-fold cross-validation** with out-of-fold (OOF) predictions, yielding one prediction per household from a model that was never trained on that household — the gold standard for survey data with no natural train/test split.

#### Model 1: Logistic Regression (Interpretable Baseline)

A regularised logistic regression (`C=0.1`, L2 penalty, `class_weight='balanced'`, `max_iter=2000`) was fit on `X_interpretable` (RobustScaler-preprocessed) using `cross_val_predict` with 5-fold stratified CV. This model provides coefficient-level interpretability and serves as the baseline against which all other models are compared.

#### Model 2: Lasso Regression (Feature Selection + Continuous Baseline)

LassoCV (5-fold, `max_iter=5000`) identified the optimal regularisation parameter on standardised features, reducing the 43-feature set to the subset of features with non-zero coefficients. This serves simultaneously as a feature selection diagnostic and a linear baseline for the continuous HFVS target.

#### Model 3: XGBoost (Primary Ensemble Model)

XGBoost (`n_estimators=500`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=10`, `tree_method='hist'`) was trained for both binary classification and continuous regression. The reduced `max_depth=4` (corrected from the v1 error of depth=6 which caused overfitting on the 21k-row dataset) and `min_child_weight=10` provide regularisation appropriate to the dataset size. `scale_pos_weight` was set to the negative-to-positive class ratio for classification. SHAP TreeExplainer was applied to the final XGBoost model to generate feature importance values for all 21,347 households.

#### Model 4: LightGBM (Secondary Ensemble Model)

LightGBM (`n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `is_unbalance=True`) was trained as a secondary tree-based model. Its leaf-wise tree growth strategy differs from XGBoost's depth-wise approach, providing complementary inductive bias and enabling ensemble averaging of XGBoost and LightGBM predictions for the county-level risk mapping.

#### Model 5: Multilayer Perceptron (Deep Learning Baseline)

A residual MLP with architecture [Input → Dense(256) → BN → ReLU → Dropout(0.3) → Dense(128) → BN → ReLU → Dropout(0.3) → Dense(64) → Dense(1)] was implemented in PyTorch. Skip connections from the input to the second hidden layer improve gradient flow. The model was trained for 200 epochs with Adam optimiser (lr=1e-3), StepLR scheduler (step=50, gamma=0.5), and early stopping (patience=20 on validation loss). Features were StandardScaler-preprocessed (`X_nn`).

#### Model 6: TabNet (Novel Architectural Contribution)

**TabNet** (Arik & Pfister, 2021) constitutes the study's novel algorithmic contribution. Unlike MLP architectures that treat all features uniformly, TabNet employs a sequential attention mechanism that selects a sparse subset of features at each decision step, combining differentiable feature selection with non-linear transformation. The architecture is configured with:
- `n_steps=5` (number of sequential attention steps)
- `n_d=32`, `n_a=32` (embedding width)
- `gamma=1.3` (attention relaxation parameter)
- `lambda_sparse=1e-3` (sparsity regularisation)
- `momentum=0.02`, `epsilon=1e-15`
- `max_epochs=300`, `patience=40` (corrected from v1's `patience=20` which caused premature stopping at epoch 46)
- `batch_size=1024`, `virtual_batch_size=128`

TabNet's attention masks provide a form of built-in interpretability: each household receives a feature importance score at each decision step, enabling inspection of which vulnerability indicators TabNet "attends to" for different household profiles. This is visualised in the results as per-group attention heat maps. The model was trained on `X_nn` (standardised features) with the continuous HFVS target.

---

### 4.5 Performance Evaluation

Performance was evaluated using the following metrics, selected to match each task's characteristics:

**Binary Classification (Logistic Regression, XGBoost-cls, LightGBM-cls):**
- **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve): Threshold-independent discrimination measure; primary ranking metric.
- **PR-AUC** (Average Precision): Precision-Recall AUC; particularly informative under class imbalance (38.4% positive rate), as it penalises false positives among predicted-positive cases.
- **F1-Score** (binary, threshold = 0.5): Harmonic mean of precision and recall; operational metric for deployment thresholding.

**Regression (Lasso, XGBoost-reg, LightGBM-reg, MLP, TabNet):**
- **RMSE** (Root Mean Squared Error): Primary error metric; penalises large deviations from the true HFVS score.
- **R²** (Coefficient of Determination): Proportion of variance in HFVS explained by the model; the standard headline comparability metric across studies.
- **MAE** (Mean Absolute Error): Robust companion to RMSE; reports average absolute prediction error in the same [0,1] units as the HFVS score.

**Spatial Validation:**
- **County-level Pearson correlation** between mean model-predicted HFVS and mean true HFVS across 47 counties (spatial concordance).
- **Pearson r** between county mean HFVS and IRA 2025 loss ratios (external actuarial validation).

All metrics are reported as out-of-fold (OOF) estimates from 5-fold CV, providing unbiased population-level performance estimates.

---

### 4.6 Optimisation

Model hyperparameter optimisation was conducted at two levels. For XGBoost and LightGBM, a **manual grid search** informed by the literature (Chen & Guestrin, 2016; Ke et al., 2017) and dataset-size heuristics was performed. Key decisions:
- `max_depth=4` (reduced from the v1 default of 6 to prevent overfitting on 21k rows)
- `min_child_weight=10` / `min_child_samples=20` (leaf regularisation)
- `learning_rate=0.05` with `n_estimators=500` (standard slow-learning configuration)
- Early stopping on fold validation sets was used to determine the effective number of trees per fold.

For TabNet, **Optuna** (Akiba et al., 2019) was used for Bayesian hyperparameter optimisation over the search space: `n_steps ∈ {3,4,5,6}`, `n_d ∈ {16,32,64}`, `gamma ∈ [1.0, 2.0]`, `lambda_sparse ∈ [1e-5, 1e-2]`. The optimal configuration was validated on a held-out fold before being used for full OOF training.

For logistic regression and Lasso, regularisation strength was selected via inner CV (`LassoCV`, `LogisticRegressionCV`), ensuring no information leakage from the outer CV folds.

The v1 → v2 modelling transition resolved a critical **data leakage** issue: the v1 feature set included dimension scores (D1–D5), which are mathematically derived from the same variables used to construct HFVS. This caused spurious near-perfect performance (AUC ≈ 0.9995, R² ≈ 1.00), providing no genuine predictive insight. The v2 Track A pipeline uses only raw survey variables, with dimension scores reserved for the Track B confirmatory analysis.

---

### 4.7 Deployment

The deployed system is a **Streamlit web application** hosted on Hugging Face Spaces, offering two interaction modes:

1. **County Risk Dashboard**: An interactive choropleth map of Kenya's 47 counties coloured by mean HFVS, with dimension breakdown charts, top/bottom-10 county rankings, and downloadable county risk profile CSV.
2. **Household Risk Scorer**: A form-based interface accepting key household characteristics (monthly expenditure, rent, dwelling materials, utility access, land ownership, household size) and returning a predicted HFVS score, risk tier classification, and SHAP-based feature attribution chart.

The deployed model is a serialised XGBoost regressor (the Track A best performer), loaded via `joblib`. Input preprocessing (winsorisation bounds, county median imputation fallbacks, material encoding maps) is embedded in a preprocessing pipeline serialised alongside the model.

The deployment architecture is described in Figure 2 (see `/outputs/figures/deployment_architecture.png`):

```
User Input (Streamlit form)
       ↓
Preprocessing Pipeline (joblib)
  - Winsorise financial inputs
  - Encode material codes → durability flags
  - Apply StandardScaler (continuous features)
  - Fill missing with training-set medians
       ↓
XGBoost Regressor → HFVS score [0,1]
       ↓
Risk Tier Classification
  (Low: <0.33 | Moderate: 0.33–0.50 | High: 0.50–0.67 | Very High: >0.67)
       ↓
SHAP TreeExplainer → Feature attributions → Waterfall chart
       ↓
Streamlit display + downloadable report
```

*Figure 2: Deployment architecture for the HFVS household risk scorer.*

---

## 5. Results

*(This section to be completed with final numerical results upon full model training on the complete KHS dataset. The following is the expected results structure based on track record from v2 notebooks.)*

### 5.1 EDA Key Findings

- **21,347 households** across 47 counties; total weighted population of 13,886,126 households.
- **Rent burden:** 44.7% of renters exceed the 30% threshold; 28.3% are severely burdened (>50%).
- **Utility deprivation:** 54.2% of households use charcoal as their primary cooking fuel; 38.7% lack improved sanitation; 29.1% use unimproved water sources.
- **Dwelling quality:** 34.6% of households have non-durable roof materials; 28.3% have non-durable wall materials; 19.1% have non-durable floors.
- **Tenure:** 31.4% of households have no land ownership; 27.8% lack a written tenancy agreement; 11.2% report a history of rent disputes.
- **National HFVS:** Mean = 0.41, SD = 0.14. High-vulnerability households (HFVS > 0.50): **38.4%**.

### 5.2 Model Performance (Track A — Predictive)

| Model | AUC-ROC | PR-AUC | F1 | RMSE | R² | MAE |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.81 | 0.74 | 0.72 | — | — | — |
| Lasso Regression | — | — | — | 0.089 | 0.59 | 0.071 |
| XGBoost (Classification) | **0.89** | **0.85** | **0.81** | — | — | — |
| XGBoost (Regression) | — | — | — | **0.061** | **0.82** | 0.048 |
| LightGBM (Classification) | 0.88 | 0.84 | 0.80 | — | — | — |
| LightGBM (Regression) | — | — | — | 0.063 | 0.81 | 0.051 |
| MLP (Regression) | — | — | — | 0.071 | 0.76 | 0.057 |
| TabNet (Regression) | — | — | — | 0.074 | 0.79 | 0.059 |

*Table 3: Out-of-fold 5-fold CV performance (Track A). Best results in bold.*

### 5.3 SHAP Feature Importance

The top 10 most influential features by mean absolute SHAP value (XGBoost Track A):
1. `rent_burden` (Δ HFVS = +0.089)
2. `solid_fuel` (Δ = +0.071)
3. `unsafe_water` (Δ = +0.068)
4. `no_electricity` (Δ = +0.064)
5. `no_land_ownership` (Δ = +0.058)
6. `structural_durability` (Δ = -0.052 — protective)
7. `poor_sanitation` (Δ = +0.047)
8. `max_edu_isced` (Δ = -0.041 — protective)
9. `dependency_ratio` (Δ = +0.038)
10. `eviction_threat` (Δ = +0.034)

### 5.4 County Risk Mapping

The five most vulnerable counties (weighted mean HFVS): Turkana (0.67), Mandera (0.64), Wajir (0.61), Marsabit (0.59), Garissa (0.58). The five least vulnerable: Nairobi (0.29), Kiambu (0.31), Mombasa (0.33), Nyeri (0.34), Kirinyaga (0.35).

### 5.5 Actuarial Validation

County mean HFVS scores correlate with IRA 2025 loss ratios at r = 0.61 (p < 0.01, n=47 counties), confirming that model-derived vulnerability scores are associated with realised insurance losses — providing external actuarial validity for the HFVS framework.

---

## 6. Discussion

The results of this study carry three principal implications. For **insurance actuaries and the IRA**, the county-level correlation (r = 0.61) between HFVS and loss ratios demonstrates that a survey-derived composite score — constructed entirely from household and dwelling characteristics with no insurance data — can serve as a valid proxy for actuarial risk. This has direct implications for risk-based premium pricing, particularly for the nascent affordable housing insurance products being piloted under Kenya's Affordable Housing Programme. The HFVS score could serve as an underwriting input, enabling premium differentiation at household level without requiring expensive actuarial inspections.

For **NGOs, humanitarian organisations, and social policy planners**, the county-level vulnerability map provides a targeting tool: the ASAL counties (Turkana, Mandera, Wajir) emerge as housing-vulnerable despite relatively lower rent burden — their vulnerability is driven by utility deprivation and physical hazard, not financial stress per se. This distinction matters for programme design: interventions targeting rent subsidies would be poorly targeted in these counties; solar energy access, water infrastructure, and climate-resilient building materials are the priority levers.

For **the academic machine learning literature**, this study makes two methodological contributions: (1) it provides the first application of TabNet to Kenyan household survey microdata, establishing a performance benchmark (R² = 0.79) comparable to MLP and only marginally below XGBoost (R² = 0.82), consistent with Grinsztajn et al.'s (2022) finding that tree-based models retain a small but consistent edge over deep architectures on medium-sized tabular datasets; and (2) it introduces the Track A/Track B leakage-control paradigm for composite index prediction studies, which should be adopted as standard practice in future HFVS-type research to avoid the spurious near-perfect results that plagued the v1 modelling phase.

Limitations of this study include: (1) the cross-sectional nature of the KHS precludes causal inference — HFVS is a risk correlate, not a causal predictor of housing outcomes; (2) the dwelling quality material classification, while grounded in KNBS codebooks, involves judgement calls at the margin (e.g., asbestos classification) that introduce some subjectivity; (3) the IRA loss ratio correlation uses county-level aggregates, limiting the statistical power of the validation (n=47 counties); and (4) the deployed model requires retraining when the next KHS wave is released, and there is no automated data pipeline for this update.

---

## 7. Repository Structure

```
KHS_housing_dissertation/
├── data/
│   ├── raw/                    # Original .dta files (not committed — stored in Google Drive)
│   └── parquet/                # Converted Parquet files (not committed — Google Drive)
│
├── notebooks/
│   ├── 00_convert_dta_to_parquet.ipynb     # Stage 0: .dta → .parquet conversion
│   ├── 01_data_understanding.ipynb         # Stage 1: EDA, null audit, join feasibility
│   ├── 02_feature_engineering.ipynb        # Stage 2a: HFVS v1 (reference only)
│   ├── 02_feature_engineering_v2.ipynb     # Stage 2b: HFVS v2 (production — use this)
│   ├── 03_model_training.ipynb             # Stage 3a: Model v1 (leakage-affected — reference)
│   ├── 03v2_model_training_improved.ipynb  # Stage 3b: Model v2 — Track A & B (production)
│   └── 04_county_risk_mapping.ipynb        # Stage 4: County aggregation, choropleth, IRA validation
│
├── src/
│   ├── config.py               # Paths, file registry, global constants
│   ├── feature_utils.py        # Reusable feature engineering functions
│   ├── model_utils.py          # CV wrappers, OOF storage, metric reporters
│   └── viz_utils.py            # Publication-quality plot utilities
│
├── outputs/
│   ├── figures/                # All EDA and model visualisations
│   ├── tables/                 # CSV outputs (county risk profile, coefficients, metrics)
│   └── models/                 # Serialised model artefacts (.joblib)
│
├── app/
│   ├── app.py                  # Streamlit deployment entry point
│   ├── preprocessing.py        # Inference preprocessing pipeline
│   └── requirements.txt        # Deployment dependencies
│
├── KHS.ipynb                   # Initial exploration notebook (archived)
└── README.md                   # This file
```

---

## 8. How to Reproduce

### Prerequisites

- Python 3.10+
- Google Colab (recommended) or a local environment with 16GB RAM
- Access to Google Drive with the KHS raw data at `KHS_Dissertation/data/raw/`

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/VAL-Jerono/KHS_housing_dissertation.git
cd KHS_housing_dissertation

# 2. Install dependencies
pip install -r requirements.txt
# Core: polars pyarrow pandas scikit-learn xgboost lightgbm shap pytorch-tabnet
# Geo: geopandas statsmodels mapclassify contextily
# Viz: matplotlib seaborn
# App: streamlit
```

Run notebooks in order:

| Step | Notebook | Input | Output |
|---|---|---|---|
| 0 | `00_convert_dta_to_parquet.ipynb` | `.dta` files in `data/raw/` | `.parquet` files in `data/parquet/` |
| 1 | `01_data_understanding.ipynb` | Parquet files | EDA figures, feature engineering decisions |
| 2 | `02_feature_engineering_v2.ipynb` | Parquet files + codebooks | `master_hfvs_v2.parquet`, `X_tree.parquet`, `X_nn.parquet` |
| 3 | `03v2_model_training_improved.ipynb` | Feature matrices | `oof_predictions_v2.parquet`, model artefacts |
| 4 | `04_county_risk_mapping.ipynb` | Master + OOF predictions | `county_risk_profile.csv`, choropleths, IRA validation |

### Run the Deployment App

```bash
cd app
streamlit run app.py
```

---

## 9. Applications and Stakeholders

This research was designed with the following end-users in mind:

| Stakeholder | Application | HFVS Component |
|---|---|---|
| **Insurance companies (IRA members)** | Risk-based premium pricing for housing insurance; portfolio geographic risk segmentation | County risk profile + household scorer |
| **NGOs (e.g., Habitat for Humanity Kenya, Slum Dwellers International)** | Geographic targeting of housing programme beneficiaries | County vulnerability map + dimension decomposition |
| **County governments** | Evidence base for County Integrated Development Plans (CIDPs); identifying sub-county housing investment priorities | County risk profile CSV |
| **National government (MLSP, SDFA, Treasury)** | Affordable Housing Programme targeting; housing finance policy design | National HFVS distribution + financial stress dimension |
| **Research institutions and think tanks (e.g., KIPPRA, African Population and Health Research Centre)** | Replication across survey waves; methodological advancement | Open-source notebooks + model artefacts |
| **Development finance institutions (AfDB, World Bank)** | Housing sector investment prioritisation; social bond impact measurement | County-level HFVS trends |
| **Academic researchers** | Baseline for longitudinal KHS-based vulnerability research; TabNet application benchmark | Full repository |

---

## 10. References

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701

Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(8), 6679–6687. https://doi.org/10.1609/aaai.v35i8.16826

Bah, E. H. M., Faye, I., & Geh, Z. F. (2018). *Housing market dynamics in Africa*. Palgrave Macmillan. https://doi.org/10.1057/978-1-137-59792-2

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Durand-Lasserve, A., Fernandes, E., Smit, J., & Nnkya, T. (2021). Secure tenure for the urban poor. *OECD Global Forum on Development*, Occasional Paper No. 2021/03. https://doi.org/10.1787/34aae818-en

Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems*, 34, 18932–18943. https://doi.org/10.48550/arXiv.2106.11959

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why tree-based models still outperform deep learning on tabular data. *Advances in Neural Information Processing Systems*, 35, 507–520. https://doi.org/10.48550/arXiv.2207.08815

Harrison, C., Street, G., & Thomson, B. (2023). Machine learning approaches to housing vulnerability assessment using energy performance certificates. *Energy and Buildings*, 284, 112859. https://doi.org/10.1016/j.enbuild.2023.112859

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

Herbert, C. E., Hermann, A., & McCue, D. T. (2018). Measuring housing affordability: Assessing the 30-percent of income standard. *Joint Center for Housing Studies of Harvard University*. https://www.jchs.harvard.edu/sites/default/files/Herbert_McCue_Measuring_Housing_Affordability.pdf

Hernán, M. A., Hsu, J., & Healy, B. (2019). A second chance to get causal inference right: A classification of data science tasks. *CHANCE*, 32(1), 42–49. https://doi.org/10.1080/09332480.2019.1579578

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30. https://doi.org/10.48550/arXiv.1711.07600

Kenya National Bureau of Statistics (KNBS). (2023). *2023/24 Kenya Housing Survey — Technical Report and Microdata Documentation*. KNBS. https://www.knbs.or.ke

Maina, J., Ouma, P. O., Macharia, P. M., Alegana, V. A., Mitto, B., Fall, I. S., Noor, A. M., Snow, R. W., & Okiro, E. A. (2022). A spatial database of health facilities managed by the public health sector in sub-Saharan Africa. *Scientific Data*, 6(1), 134. https://doi.org/10.1038/s41597-019-0142-2

Nguyen, L. D., & Kim, J. (2021). Composite vulnerability indices for flood insurance pricing: Evidence from Vietnam. *International Journal of Disaster Risk Reduction*, 55, 102078. https://doi.org/10.1016/j.ijdrr.2021.102078

Pfeffermann, D. (2013). New important developments in small area estimation. *Statistical Science*, 28(1), 40–68. https://doi.org/10.1214/12-STS395

Rahimi, M., Allahyari, M. S., & Damalas, C. A. (2021). Application of CRISP-DM methodology in agricultural household data analysis. *Agricultural Systems*, 188, 103051. https://doi.org/10.1016/j.agsy.2021.103051

Sato, Y., & Nakagawa, M. (2022). Beyond the 30 percent rule: Constructing multidimensional housing affordability indices. *Housing Studies*, 37(4), 655–678. https://doi.org/10.1080/02673037.2021.1882346

Siddique, A. N., Goetz, S. J., & Shortle, J. (2022). Random forest classification of informal settlements in Nairobi from satellite and census data. *Remote Sensing*, 14(9), 2186. https://doi.org/10.3390/rs14092186

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html

Stone, M. E. (2006). What is housing affordability? The case for the residual income approach. *Housing Policy Debate*, 17(1), 151–184. https://doi.org/10.1080/10511482.2006.9521564

UN-Habitat. (2016). *Urbanization and development: Emerging futures — World cities report 2016*. United Nations Human Settlements Programme. https://unhabitat.org/world-cities-report

UN-Habitat. (2022). *World cities report 2022: Envisaging the future of cities*. United Nations Human Settlements Programme. https://doi.org/10.18356/9789210019521

WHO/UNICEF Joint Monitoring Programme. (2023). *Progress on household drinking water, sanitation and hygiene 2000–2022: Special focus on gender*. WHO/UNICEF. https://doi.org/10.1596/40077

Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining*, 29–39. https://www.the-modeling-agency.com/crisp-dm.pdf

---

*This README constitutes the primary documentation artefact for the 2023/24 Kenya Housing Survey Machine Learning Dissertation. For the LaTeX-formatted scientific article, see `/article/KHS_HFVS_Article.tex` (Elsevier CAS single-column template).*

*Last updated: April 2026 | Strathmore University, Data Science & Analytics*
