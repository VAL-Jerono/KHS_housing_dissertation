// Kenya Housing Survey 2023/24 · HFVS Research Data
// Strathmore University MSc Dissertation

export const NATIONAL = {
  mean: 0.3236,
  median: 0.3246,
  sd: 0.0313,
  range: [0.2751, 0.4027] as [number, number],
  n_households: 21347,
  n_counties: 47,
  pct_high_vuln: 0.40,
  pct_urban: 0.443,
};

export type County = {
  id: number;
  name: string;
  hfvs: number;
  d1: number; // Financial stress (weight 0.30)
  d2: number; // Tenure insecurity (weight 0.25)
  d3: number; // Physical hazard (weight 0.25)
  d4: number; // Dwelling quality (weight 0.20)
  d5: number; // Utility deprivation (weight 0.20)
  pct_high: number;
  pct_urban: number;
  n_households: number;
};

export const COUNTIES: County[] = [
  { id: 4,  name: "Tana River",       hfvs: 0.403, d1: 0.390, d2: 0.231, d3: 0.389, d4: 0.310, d5: 0.348, pct_high: 0.74, pct_urban: 0.47, n_households: 450 },
  { id: 26, name: "Trans Nzoia",      hfvs: 0.384, d1: 0.410, d2: 0.260, d3: 0.220, d4: 0.320, d5: 0.381, pct_high: 0.68, pct_urban: 0.39, n_households: 373 },
  { id: 24, name: "West Pokot",       hfvs: 0.375, d1: 0.350, d2: 0.310, d3: 0.350, d4: 0.290, d5: 0.421, pct_high: 0.65, pct_urban: 0.28, n_households: 312 },
  { id: 25, name: "Samburu",          hfvs: 0.371, d1: 0.330, d2: 0.350, d3: 0.310, d4: 0.270, d5: 0.441, pct_high: 0.63, pct_urban: 0.22, n_households: 287 },
  { id: 8,  name: "Wajir",            hfvs: 0.368, d1: 0.320, d2: 0.280, d3: 0.290, d4: 0.250, d5: 0.472, pct_high: 0.61, pct_urban: 0.31, n_households: 334 },
  { id: 9,  name: "Mandera",          hfvs: 0.365, d1: 0.310, d2: 0.270, d3: 0.280, d4: 0.240, d5: 0.481, pct_high: 0.59, pct_urban: 0.28, n_households: 298 },
  { id: 23, name: "Turkana",          hfvs: 0.361, d1: 0.300, d2: 0.290, d3: 0.310, d4: 0.260, d5: 0.462, pct_high: 0.57, pct_urban: 0.25, n_households: 342 },
  { id: 7,  name: "Garissa",          hfvs: 0.358, d1: 0.320, d2: 0.260, d3: 0.270, d4: 0.220, d5: 0.452, pct_high: 0.56, pct_urban: 0.33, n_households: 289 },
  { id: 10, name: "Marsabit",         hfvs: 0.352, d1: 0.290, d2: 0.300, d3: 0.280, d4: 0.240, d5: 0.431, pct_high: 0.54, pct_urban: 0.27, n_households: 276 },
  { id: 11, name: "Isiolo",           hfvs: 0.347, d1: 0.310, d2: 0.280, d3: 0.250, d4: 0.220, d5: 0.411, pct_high: 0.52, pct_urban: 0.41, n_households: 245 },
  { id: 2,  name: "Kwale",            hfvs: 0.344, d1: 0.330, d2: 0.240, d3: 0.280, d4: 0.240, d5: 0.362, pct_high: 0.51, pct_urban: 0.34, n_households: 412 },
  { id: 43, name: "Homa Bay",         hfvs: 0.341, d1: 0.340, d2: 0.220, d3: 0.240, d4: 0.250, d5: 0.341, pct_high: 0.49, pct_urban: 0.27, n_households: 435 },
  { id: 44, name: "Migori",           hfvs: 0.338, d1: 0.330, d2: 0.210, d3: 0.250, d4: 0.240, d5: 0.332, pct_high: 0.48, pct_urban: 0.29, n_households: 418 },
  { id: 40, name: "Busia",            hfvs: 0.336, d1: 0.320, d2: 0.230, d3: 0.220, d4: 0.250, d5: 0.341, pct_high: 0.47, pct_urban: 0.30, n_households: 395 },
  { id: 6,  name: "Taita Taveta",     hfvs: 0.334, d1: 0.310, d2: 0.240, d3: 0.260, d4: 0.230, d5: 0.322, pct_high: 0.46, pct_urban: 0.37, n_households: 356 },
  { id: 3,  name: "Kilifi",           hfvs: 0.332, d1: 0.320, d2: 0.230, d3: 0.270, d4: 0.220, d5: 0.311, pct_high: 0.45, pct_urban: 0.35, n_households: 498 },
  { id: 41, name: "Siaya",            hfvs: 0.330, d1: 0.330, d2: 0.210, d3: 0.230, d4: 0.230, d5: 0.321, pct_high: 0.44, pct_urban: 0.26, n_households: 424 },
  { id: 38, name: "Vihiga",           hfvs: 0.328, d1: 0.320, d2: 0.220, d3: 0.210, d4: 0.240, d5: 0.331, pct_high: 0.43, pct_urban: 0.28, n_households: 367 },
  { id: 15, name: "Kitui",            hfvs: 0.327, d1: 0.300, d2: 0.240, d3: 0.240, d4: 0.230, d5: 0.322, pct_high: 0.42, pct_urban: 0.29, n_households: 448 },
  { id: 17, name: "Makueni",          hfvs: 0.325, d1: 0.300, d2: 0.230, d3: 0.230, d4: 0.220, d5: 0.312, pct_high: 0.41, pct_urban: 0.28, n_households: 412 },
  { id: 39, name: "Bungoma",          hfvs: 0.325, d1: 0.310, d2: 0.210, d3: 0.200, d4: 0.240, d5: 0.331, pct_high: 0.41, pct_urban: 0.31, n_households: 456 },
  { id: 37, name: "Kakamega",         hfvs: 0.324, d1: 0.310, d2: 0.200, d3: 0.190, d4: 0.240, d5: 0.332, pct_high: 0.41, pct_urban: 0.32, n_households: 523 },
  { id: 42, name: "Kisumu",           hfvs: 0.323, d1: 0.320, d2: 0.190, d3: 0.210, d4: 0.220, d5: 0.291, pct_high: 0.40, pct_urban: 0.54, n_households: 445 },
  { id: 36, name: "Bomet",            hfvs: 0.322, d1: 0.290, d2: 0.210, d3: 0.200, d4: 0.220, d5: 0.301, pct_high: 0.39, pct_urban: 0.27, n_households: 389 },
  { id: 16, name: "Machakos",         hfvs: 0.320, d1: 0.300, d2: 0.200, d3: 0.190, d4: 0.210, d5: 0.291, pct_high: 0.38, pct_urban: 0.42, n_households: 467 },
  { id: 35, name: "Kericho",          hfvs: 0.320, d1: 0.290, d2: 0.200, d3: 0.190, d4: 0.220, d5: 0.281, pct_high: 0.37, pct_urban: 0.35, n_households: 378 },
  { id: 46, name: "Nyamira",          hfvs: 0.319, d1: 0.290, d2: 0.210, d3: 0.190, d4: 0.220, d5: 0.281, pct_high: 0.37, pct_urban: 0.22, n_households: 345 },
  { id: 18, name: "Nyandarua",        hfvs: 0.318, d1: 0.280, d2: 0.220, d3: 0.200, d4: 0.210, d5: 0.271, pct_high: 0.36, pct_urban: 0.23, n_households: 356 },
  { id: 33, name: "Narok",            hfvs: 0.316, d1: 0.280, d2: 0.230, d3: 0.210, d4: 0.200, d5: 0.271, pct_high: 0.35, pct_urban: 0.31, n_households: 378 },
  { id: 14, name: "Embu",             hfvs: 0.315, d1: 0.280, d2: 0.210, d3: 0.190, d4: 0.200, d5: 0.261, pct_high: 0.35, pct_urban: 0.39, n_households: 334 },
  { id: 12, name: "Meru",             hfvs: 0.314, d1: 0.280, d2: 0.200, d3: 0.180, d4: 0.210, d5: 0.261, pct_high: 0.34, pct_urban: 0.38, n_households: 489 },
  { id: 28, name: "Elgeyo Marakwet",  hfvs: 0.313, d1: 0.270, d2: 0.220, d3: 0.200, d4: 0.200, d5: 0.251, pct_high: 0.34, pct_urban: 0.28, n_households: 298 },
  { id: 13, name: "Tharaka Nithi",    hfvs: 0.312, d1: 0.270, d2: 0.210, d3: 0.190, d4: 0.200, d5: 0.251, pct_high: 0.33, pct_urban: 0.25, n_households: 267 },
  { id: 30, name: "Baringo",          hfvs: 0.312, d1: 0.270, d2: 0.220, d3: 0.200, d4: 0.190, d5: 0.251, pct_high: 0.33, pct_urban: 0.30, n_households: 312 },
  { id: 29, name: "Nandi",            hfvs: 0.311, d1: 0.270, d2: 0.210, d3: 0.190, d4: 0.190, d5: 0.241, pct_high: 0.32, pct_urban: 0.28, n_households: 334 },
  { id: 22, name: "Kiambu",           hfvs: 0.311, d1: 0.280, d2: 0.190, d3: 0.170, d4: 0.180, d5: 0.221, pct_high: 0.31, pct_urban: 0.62, n_households: 534 },
  { id: 31, name: "Laikipia",         hfvs: 0.310, d1: 0.270, d2: 0.200, d3: 0.180, d4: 0.190, d5: 0.231, pct_high: 0.32, pct_urban: 0.36, n_households: 312 },
  { id: 45, name: "Kisii",            hfvs: 0.309, d1: 0.280, d2: 0.200, d3: 0.170, d4: 0.190, d5: 0.221, pct_high: 0.31, pct_urban: 0.24, n_households: 456 },
  { id: 27, name: "Uasin Gishu",      hfvs: 0.308, d1: 0.270, d2: 0.180, d3: 0.170, d4: 0.180, d5: 0.231, pct_high: 0.30, pct_urban: 0.47, n_households: 389 },
  { id: 5,  name: "Lamu",             hfvs: 0.307, d1: 0.270, d2: 0.190, d3: 0.210, d4: 0.180, d5: 0.211, pct_high: 0.30, pct_urban: 0.41, n_households: 198 },
  { id: 21, name: "Muranga",          hfvs: 0.305, d1: 0.260, d2: 0.190, d3: 0.160, d4: 0.180, d5: 0.211, pct_high: 0.29, pct_urban: 0.25, n_households: 412 },
  { id: 32, name: "Nakuru",           hfvs: 0.303, d1: 0.260, d2: 0.180, d3: 0.160, d4: 0.170, d5: 0.201, pct_high: 0.28, pct_urban: 0.51, n_households: 521 },
  { id: 1,  name: "Mombasa",          hfvs: 0.302, d1: 0.280, d2: 0.170, d3: 0.180, d4: 0.160, d5: 0.171, pct_high: 0.27, pct_urban: 0.93, n_households: 423 },
  { id: 34, name: "Kajiado",          hfvs: 0.299, d1: 0.260, d2: 0.180, d3: 0.160, d4: 0.170, d5: 0.191, pct_high: 0.26, pct_urban: 0.38, n_households: 356 },
  { id: 20, name: "Kirinyaga",        hfvs: 0.298, d1: 0.250, d2: 0.180, d3: 0.150, d4: 0.170, d5: 0.191, pct_high: 0.25, pct_urban: 0.31, n_households: 334 },
  { id: 19, name: "Nyeri",            hfvs: 0.293, d1: 0.250, d2: 0.180, d3: 0.140, d4: 0.160, d5: 0.181, pct_high: 0.24, pct_urban: 0.41, n_households: 378 },
  { id: 47, name: "Nairobi",          hfvs: 0.283, d1: 0.260, d2: 0.160, d3: 0.140, d4: 0.130, d5: 0.151, pct_high: 0.19, pct_urban: 1.00, n_households: 678 },
];

export const MODELS = {
  classification: [
    { name: "LightGBM",     auc: 0.9892, pr_auc: 0.9853, f1: 0.9289, color: "#4A8EE0" },
    { name: "XGBoost",      auc: 0.9888, pr_auc: 0.9851, f1: 0.9286, color: "#35A882" },
    { name: "Logistic Reg", auc: 0.9832, pr_auc: 0.9773, f1: 0.9072, color: "#F2A836" },
  ],
  regression: [
    { name: "XGBoost",  r2: 0.9696, rmse: 0.01511, mae: 0.01095, color: "#35A882" },
    { name: "LightGBM", r2: 0.9695, rmse: 0.01512, mae: 0.01093, color: "#4A8EE0" },
    { name: "MLP",      r2: 0.9530, rmse: 0.01879, mae: 0.01410, color: "#8B7BE8" },
    { name: "TabNet",   r2: 0.9316, rmse: 0.02266, mae: 0.01608, color: "#F2A836" },
  ],
};

export const TOP_FEATURES = [
  { name: "savings_rate",          mi: 0.142, dim: "financial" },
  { name: "no_savings",            mi: 0.138, dim: "financial" },
  { name: "structural_durability", mi: 0.124, dim: "dwelling"  },
  { name: "no_land_ownership",     mi: 0.108, dim: "tenure"    },
  { name: "informal_dwelling",     mi: 0.090, dim: "dwelling"  },
  { name: "floor_durable",         mi: 0.081, dim: "dwelling"  },
  { name: "county_hfvs_rank",      mi: 0.079, dim: "spatial"   },
  { name: "wall_durable",          mi: 0.072, dim: "dwelling"  },
  { name: "roof_durable",          mi: 0.068, dim: "dwelling"  },
  { name: "log_rent",              mi: 0.063, dim: "financial" },
  { name: "rent_burden",           mi: 0.058, dim: "financial" },
  { name: "no_written_lease",      mi: 0.054, dim: "tenure"    },
  { name: "eviction_threat",       mi: 0.049, dim: "tenure"    },
  { name: "near_flood_zone",       mi: 0.044, dim: "hazard"    },
  { name: "no_electricity",        mi: 0.039, dim: "utility"   },
];

export const DIM_COLORS: Record<string, string> = {
  financial: "#F2A836",
  tenure:    "#E25A38",
  dwelling:  "#8B7BE8",
  hazard:    "#4A8EE0",
  utility:   "#35A882",
  spatial:   "#7A8595",
};

export const DIMENSIONS = [
  { key: "d1", label: "Financial Stress",    color: "#F2A836", weight: 0.30, desc: "Rent burden, savings capacity, expenditure barriers" },
  { key: "d2", label: "Tenure Insecurity",   color: "#E25A38", weight: 0.25, desc: "Eviction risk, land ownership, legal protection" },
  { key: "d3", label: "Physical Hazard",     color: "#4A8EE0", weight: 0.25, desc: "Flood, mudslide, hazard proximity (enumerator-observed)" },
  { key: "d4", label: "Dwelling Quality",    color: "#8B7BE8", weight: 0.20, desc: "Wall, floor, roof construction materials" },
  { key: "d5", label: "Utility Deprivation", color: "#35A882", weight: 0.20, desc: "Electricity, water, sanitation access" },
];

export function getRiskTier(hfvs: number): "HIGH" | "ABOVE MEAN" | "BELOW MEAN" {
  if (hfvs > NATIONAL.mean + NATIONAL.sd) return "HIGH";
  if (hfvs > NATIONAL.mean) return "ABOVE MEAN";
  return "BELOW MEAN";
}

export function getRiskColor(hfvs: number): string {
  if (hfvs > NATIONAL.mean + NATIONAL.sd) return "#E25A38";
  if (hfvs > NATIONAL.mean) return "#F2A836";
  return "#35A882";
}

export function getCountyById(id: number): County | undefined {
  return COUNTIES.find((c) => c.id === id);
}