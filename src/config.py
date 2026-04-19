import os
from pathlib import Path

# ── Google Drive paths (used in Colab) ──────────────────────────────
DRIVE_ROOT = Path("/content/drive/MyDrive/KHS_Dissertation")
RAW_DATA   = DRIVE_ROOT / "data" / "raw"
PARQUET    = DRIVE_ROOT / "data" / "parquet"
OUTPUTS    = DRIVE_ROOT / "outputs"

# ── Data file registry ───────────────────────────────────────────────
DTA_FILES = {
    "household":   "Household_Information_Data.dta",
    "individual":  "Individual_Microdata.dta",
    "dwelling":    "Dwelling_Units_Microdata.dta",
    "mortgage":    "Housing_Mortgage.dta",
    "loan":        "Housing_Loan_Microdata.dta",
    "financiers":  "Housing_Financiers_Microdata.dta",
    "county":      "County_Microdata.dta",
    "real_estate": "Real_Estate_Microdata.dta",
    "land":        "Land_Parcels_Microdata.dta",
    "institution": "Institution_Microdata.dta",
    "housing_types":"Type_of_Housing_Units_Microdata.dta",
    "project":     "Project_Information_Microdata.dta",
    "water":       "Water_Service_Providers_Microdata.dta",
    "nema":        "NEMA_Microdata.dta",
}

def raw(key):
    return RAW_DATA / DTA_FILES[key]

def parquet(key):
    return PARQUET / DTA_FILES[key].replace(".dta", ".parquet")