# Modelling Housing-Based Financial Vulnerability and Insurance Risk  
### Kenya Housing Survey 2023/24 · MSc Data Science & Analytics · Strathmore University

## Overview
This dissertation models household-level financial vulnerability and 
housing risk across all 47 Kenyan counties using the 2023/24 KNBS 
Kenya Housing Survey microdata.

## Data
Raw data files (.dta) are sourced from the Kenya National Data Archive:  
https://www.kenada.knbs.or.ke — DDI-KEN-KNBS-KHS-2023-24-V001

Data files are **not** committed to this repository.  
After downloading, upload them to Google Drive at:  
`MyDrive/KHS_Dissertation/data/raw/`  
Then run `notebooks/00_convert_dta_to_parquet.ipynb` once.

## Setup (Google Colab)
1. Upload raw `.dta` files to Google Drive (path above)
2. Open any notebook in Colab
3. Run `00_convert_dta_to_parquet.ipynb` first (one-time)
4. All subsequent notebooks load from parquet automatically

## Notebooks
| # | Notebook | Description |
|---|----------|-------------|
| 00 | convert_dta_to_parquet | One-time DTA → parquet conversion |
| 01 | data_understanding | EDA, null audit, codebook mapping |
| 02 | feature_engineering | Risk index construction |
| 03 | model_training | XGBoost + SHAP analysis |
| 04 | county_risk_mapping | Spatial aggregation + IRA validation |

## Stack
Python · Polars · XGBoost · SHAP · Scikit-learn · Supabase (final dataset)