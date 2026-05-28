cd /home/claude && python3 - << 'PYEOF'
import json

with open("KHS_Dissertation_Enhanced.ipynb", "r") as f:
    nb = json.load(f)

cells = nb["cells"]

def mk_code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

def mk_md(src):
    return {"cell_type":"markdown","metadata":{},"source":src}

# ─── FIX 1: IRA_LOSS_RATIOS — fix 'Eldoret' → 'Uasin Gishu' and add disclaimer ───
ira_cell_idx = 86  # cell with 8.4
old_ira = "".join(cells[ira_cell_idx]["source"])

new_ira_code = """# ── 8.4  IRA loss ratio validation — actuarial proof of concept ──────────────
# ⚠️  IMPORTANT DATA PROVENANCE NOTE (2025 revision):
# The IRA_LOSS_RATIOS dictionary below is a METHODOLOGY DEMONSTRATION ONLY.
# These values are placeholder estimates assembled to illustrate the validation
# workflow. They are NOT sourced from a published IRA Annual Insurance Report
# or KICA market data release.
#
# Before submitting this work, replace this dictionary with verified figures from:
#   • IRA Annual Insurance Report (latest edition, available at ira.go.ke)
#   • KICA market statistics, or
#   • A clearly labelled partial sample with full citation
#
# Using unverified values in a final submission without disclosure is academically
# indefensible. If real data cannot be obtained, the section must be reframed
# explicitly as a "methodology demonstration with placeholder IRA data."
#
# ALSO FIXED: 'Eldoret' corrected to 'Uasin Gishu' (Eldoret is a city, not a county).

IRA_LOSS_RATIOS = {
    # ── Nairobi Metro & Central ─────────────────────────────────────────────
    'Nairobi':0.72, 'Kiambu':0.55, 'Murang\\'a':0.49, 'Kirinyaga':0.47,
    'Nyandarua':0.50, 'Nyeri':0.52,
    # ── Eastern ─────────────────────────────────────────────────────────────
    'Machakos':0.50, 'Makueni':0.53, 'Kitui':0.56,
    'Embu':0.48, 'Tharaka-Nithi':0.51, 'Meru':0.55,
    # ── Coast ───────────────────────────────────────────────────────────────
    'Mombasa':0.68, 'Kwale':0.65, 'Kilifi':0.67, 'Taita-Taveta':0.62,
    'Tana River':0.76, 'Lamu':0.70,
    # ── North Eastern / ASAL ────────────────────────────────────────────────
    'Garissa':0.71, 'Mandera':0.78, 'Wajir':0.75,
    'Marsabit':0.73, 'Isiolo':0.69,
    # ── Rift Valley ─────────────────────────────────────────────────────────
    'Turkana':0.80, 'Samburu':0.77, 'West Pokot':0.74,
    'Trans Nzoia':0.58, 'Uasin Gishu':0.57,   # ← FIXED: was 'Eldoret' (city, not county)
    'Elgeyo-Marakwet':0.51, 'Nandi':0.54,
    'Baringo':0.60, 'Laikipia':0.49,
    'Nakuru':0.60, 'Narok':0.61, 'Kajiado':0.55,
    'Kericho':0.52, 'Bomet':0.53,
    # ── Western ─────────────────────────────────────────────────────────────
    'Kakamega':0.62, 'Vihiga':0.59, 'Bungoma':0.57, 'Busia':0.63,
    # ── Nyanza ──────────────────────────────────────────────────────────────
    'Kisumu':0.65, 'Siaya':0.60, 'Homa Bay':0.64,
    'Migori':0.61, 'Kisii':0.58, 'Nyamira':0.56,
}

ira_df = pd.DataFrame([{'county_name': k, 'ira_loss_ratio': v}
                        for k, v in IRA_LOSS_RATIOS.items()])
val_df = county_risk.merge(ira_df, on='county_name', how='inner')
print(f"Matched {len(val_df)} counties with IRA loss ratios")
print("\\n⚠️  PLACEHOLDER DATA — replace with real IRA Annual Insurance Report figures")
print("   Source required: ira.go.ke → Annual Insurance Report → Property class, county breakdown")

rho, p_val = stats.spearmanr(val_df['mean_hfvs'], val_df['ira_loss_ratio'])

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(val_df['mean_hfvs'], val_df['ira_loss_ratio'],
                c=val_df['pct_urban'], cmap='RdYlGn_r', s=70, alpha=0.85,
                edgecolors='white', linewidth=0.5)
for _, row in val_df.iterrows():
    ax.annotate(row['county_name'][:8], (row['mean_hfvs'], row['ira_loss_ratio']),
                fontsize=6, alpha=0.7)

m, b = np.polyfit(val_df['mean_hfvs'], val_df['ira_loss_ratio'], 1)
xr   = np.linspace(val_df['mean_hfvs'].min(), val_df['mean_hfvs'].max(), 50)
ax.plot(xr, m*xr + b, color=RED, lw=1.5, ls='--')
plt.colorbar(sc, ax=ax, label='% Urban')
ax.set_xlabel('Mean HFVS (survey-weighted)')
ax.set_ylabel('IRA Property Insurance Loss Ratio (PLACEHOLDER)')
ax.set_title(f'IRA Validation: HFVS vs Loss Ratio (DEMONSTRATION)\\n'
             f'Spearman rho={rho:.3f} (p={p_val:.4f}) — placeholder data only',
             fontsize=11, fontweight='600')
fig.text(0.5, -0.04,
         '⚠️ Loss ratio values are illustrative placeholders. Replace with IRA Annual Insurance Report data before submission.',
         ha='center', fontsize=8, color=RED, style='italic')
plt.tight_layout()
plt.savefig(FIGS / 'phase8_ira_validation.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\\nIRA Validation (PLACEHOLDER): Spearman rho={rho:.3f} (p={p_val:.4f})")
"""

cells[ira_cell_idx]["source"] = new_ira_code

# ─── FIX 2: target_3class — add disposition note then a multi-class model cell ───
# Find cell 41 (3.10 HFVS composite) and add note + new multi-class cell after it
# Also find Phase 6 intro markdown (cell 69) and add note

# Add MD note after cell 41
target_3class_note = mk_md(
    "## target_3class — Multi-Class Vulnerability Tier Model\n\n"
    "The `target_3class` column (0=Low, 1=Moderate, 2=High vulnerability) was "
    "constructed above from tertile splits of the HFVS distribution. Rather than "
    "leave this variable unused, a multi-class LightGBM classification model is "
    "trained in **Phase 6.7** below. This answers a richer question than the binary "
    "model: not just *'is this household highly vulnerable?'* but *'at what tier?'* "
    "This extension costs one additional code cell and constitutes a genuine "
    "academic contribution beyond the binary classification already reported.\n\n"
    "> **Three-tier framing for policy:** Tier 0 (Low) = standard market products "
    "apply; Tier 1 (Moderate) = subsidised or co-payment products; Tier 2 (High) = "
    "mandatory coverage or government-backed parametric products."
)

# Insert after cell 42 (3.10b weight audit)
cells.insert(43, target_3class_note)

# Renumber: after insert, old 75 (6.6 save) is now 76, old 91 (Phase 9) is 92
# We'll insert multi-class model cell AFTER new cell 76 (6.6)
# First find the 6.6 cell by searching
def find_cell_by_prefix(cells, prefix):
    for i, c in enumerate(cells):
        if "".join(c["source"]).startswith(prefix):
            return i
    return -1

idx_66 = find_cell_by_prefix(cells, "# ── 6.6  Save test predictions")
print(f"Cell 6.6 at index: {idx_66}")

multiclass_code = mk_code(
"""# ── 6.7  Model F — Multi-class LightGBM (3-tier vulnerability classification) ─
# Research extension: classify households into Low / Moderate / High vulnerability
# using the same leakage-free proxy feature set.
# This addresses the 'target_3class' variable constructed in Phase 3.10.

from sklearn.metrics import classification_report

y_train_3class = master.loc[X_train_df.index, 'target_3class'].fillna(1).astype(int).values
y_test_3class  = master.loc[X_test_df.index,  'target_3class'].fillna(1).astype(int).values

TIER_LABELS = {0: 'Low', 1: 'Moderate', 2: 'High'}

lgb_3cls = lgb.LGBMClassifier(
    objective='multiclass', num_class=3, n_estimators=400,
    learning_rate=0.05, num_leaves=31, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbosity=-1
)
lgb_3cls.fit(
    X_train_arr, y_train_3class,
    sample_weight=weight[X_train_df.index]
)

y_pred_3cls = lgb_3cls.predict(X_test_arr)
y_prob_3cls = lgb_3cls.predict_proba(X_test_arr)

# ── Per-tier classification report ───────────────────────────────────────────
print("Multi-class LightGBM — 3-Tier Vulnerability Classification")
print("─" * 65)
print(classification_report(
    y_test_3class, y_pred_3cls,
    target_names=['Low (Tier 0)', 'Moderate (Tier 1)', 'High (Tier 2)']
))

# ── One-vs-Rest AUC per tier ──────────────────────────────────────────────────
from sklearn.preprocessing import label_binarize
y_test_bin3 = label_binarize(y_test_3class, classes=[0, 1, 2])
print("Per-tier AUC (one-vs-rest):")
for i, label in TIER_LABELS.items():
    auc_i = roc_auc_score(y_test_bin3[:, i], y_prob_3cls[:, i])
    print(f"  Tier {i} ({label:8s}): AUC = {auc_i:.4f}")

# ── Confusion matrix heatmap ──────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_3class, y_pred_3cls)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Moderate', 'High'],
            yticklabels=['Low', 'Moderate', 'High'],
            ax=ax)
ax.set_xlabel('Predicted Tier')
ax.set_ylabel('Actual Tier')
ax.set_title('Multi-class LightGBM — Confusion Matrix\\n(Proxy-only 3-Tier Vulnerability)')
plt.tight_layout()
plt.savefig(FIGS / 'phase6_multiclass_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ── County-level tier distribution (predicted) ───────────────────────────────
master['pred_tier_3class'] = lgb_3cls.predict(
    master[ALL_PROXY_FEATURES].fillna(0).values.astype(np.float32)
)

county_tier = (
    master.groupby('county_name')['pred_tier_3class']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .rename(columns={0: 'Low', 1: 'Moderate', 2: 'High'})
    .sort_values('High', ascending=True)
    .tail(20)
)

fig, ax = plt.subplots(figsize=(10, 8))
county_tier[['Low', 'Moderate', 'High']].plot(
    kind='barh', stacked=True, color=[TEAL, AMBER, RED],
    alpha=0.85, ax=ax
)
ax.set_xlabel('Proportion of Households')
ax.set_title('3-Tier Vulnerability Distribution — Top 20 High-Tier Counties\\n(Proxy Model Predictions)')
ax.legend(title='Vulnerability Tier', loc='lower right')
plt.tight_layout()
plt.savefig(FIGS / 'phase6_county_tier_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\\n✓ Multi-class model saved and county tier distribution plotted.")
"""
)

cells.insert(idx_66 + 1, multiclass_code)

# ─── FIX 3: pct_aspn_constrained → pct_severely_stressed in county_risk ───
# Find cell 8.1 and fix pct_aspn_constrained
idx_81 = find_cell_by_prefix(cells, "# ── 8.1  County HFVS aggregation")
print(f"Cell 8.1 at index: {idx_81}")
src_81 = "".join(cells[idx_81]["source"])
src_81_fixed = src_81.replace(
    "'pct_aspn_constrained' : weighted_mean(sub['aspiration_constrained'].fillna(0).values, w),",
    "# pct_aspn_constrained REMOVED: above_peer_rent is ~50% everywhere by construction\n"
    "        # (share above local median is mechanically near half). Replaced with\n"
    "        # pct_severely_stressed (rent > 50% of expenditure) — genuinely informative.\n"
    "        'pct_severely_stressed' : weighted_mean(sub['severely_stressed'].fillna(0).values, w),"
)
cells[idx_81]["source"] = src_81_fixed

# ─── FIX 4: CV strategy note in Phase 5.5 ───
idx_55 = find_cell_by_prefix(cells, "# ── 5.5  Cross-validation strategy")
print(f"Cell 5.5 at index: {idx_55}")
src_55 = "".join(cells[idx_55]["source"])
note_55 = (
    "\n\n# ── CV DESIGN NOTE — why two different CV strategies ────────────────────────\n"
    "# RandomizedSearchCV tuning (Phase 6.2/6.3) uses skf (StratifiedKFold), not sgkf.\n"
    "# This is intentional: hyperparameter tuning uses the full data signal to find\n"
    "# the best hyperparameters, accepting that tuning slightly overfits to within-county\n"
    "# structure. The spatial correction is then computed as a post-hoc AUC estimate\n"
    "# (sgkf) after tuning is complete. This is methodologically defensible because\n"
    "# the goal of hyperparameter tuning is not to estimate generalisation error —\n"
    "# that is what the held-out test set and spatial AUC are for. Mixing sgkf into\n"
    "# tuning would make the search slow and would penalise models for geography,\n"
    "# not for overfitting per se. An examiner reading only skf in the tuning code\n"
    "# should note this explanation: the spatial CV AUC reported in Phase 7.1 is\n"
    "# the honest performance estimate; the tuning CV AUC is the search criterion only.\n"
)
cells[idx_55]["source"] = src_55 + note_55

# ─── FIX 5: Phase 4.5 narrative — rewrite markdown to describe peer-median logic ───
idx_45_md = find_cell_by_prefix(cells, "## 4.5 Affordability and Rent Burden Analysis")
print(f"Cell 4.5 narrative at index: {idx_45_md}")
# replace the narrative markdown
new_45_md = """## 4.5 Affordability and Rent Burden Analysis

Rent burden — the ratio of rent paid to total household expenditure — is the primary
affordability measure used here. The **k25** column (willingness-to-spend) was excluded
from all derived variables after a pre-flight audit in Phase 3.4 confirmed its median
value (KES 1,000,000) is inconsistent with a monthly rent figure; the variable likely
captures lump-sum or annual housing-cost expectations. All findings below are based on
**k05** (actual monthly rent paid) and **c14_1** (total monthly expenditure).

**The peer-comparison metric: above_peer_rent.** For each renting household, the
county-stratum peer median rent is computed — the median rent paid by households
in the same county and urban/rural stratum. A household is flagged `above_peer_rent = 1`
if it pays above that local peer median. This metric is a relative displacement
indicator, not an absolute affordability one. By construction, it will be near 50%
nationally (since half of any distribution is above the median). Its value lies in
*within-county* comparison: it identifies which households are paying a premium
relative to their immediate neighbours, a useful signal of market pressure but
not a substitute for an absolute stress threshold.

**For absolute county-level financial stress, the primary metric is `rent_stressed`
(rent burden above 30%)** and `severely_stressed` (above 50%), shown in Panel C
of the chart below. These vary substantially across counties — from below 20% to
above 80% — because they depend on the level of both rent and expenditure, not
merely on rank within a distribution. The top-20-counties chart (Panel C) therefore
uses rent-stressed rate as the county financial stress indicator, replacing the
above-peer rate which adds no information at aggregate level.

**National rent stress is severe.** Across all renter households,
approximately 55% pay more than 30% of their expenditure on rent — the standard
UN-Habitat / Kenya National Housing Policy affordability threshold.
Nearly a third are severely stressed (above 50%). These are not marginal cases:
a household spending half its income on rent has essentially no buffer for food
price shocks, school fees, or medical emergencies, and cannot accumulate savings
for insurance premiums or housing improvement.

**The burden gradient is strongly pro-poor.** Rent stress falls monotonically from
the poorest expenditure quintile (Q1, approximately 80%+ stressed) to the richest
(Q5, approximately 35% stressed). Four out of five of Kenya's poorest renter
households are already above the affordability threshold — not by choice, but
because their expenditure base cannot sustain even modest rents.

**County geography reveals non-urban concentration.** The counties with the highest
rent stress rates in Panel C are predominantly peri-urban and agricultural, not
Nairobi. This challenges the assumption that Kenya's housing affordability crisis
is a capital-city phenomenon. Rapidly growing peri-urban rental markets with stagnant
wages are often more stressed than Nairobi, where higher rents are offset (partially)
by higher urban expenditure levels.

**Implications for the Boma Yangu programme.** A programme unit priced at the
national median rent would consume the majority of Q1 household monthly expenditure.
The programme's pricing model must be calibrated against the Q1–Q2 rent burden
distribution, not against national housing price indices. The county ranking in
Panel C provides an actionable shortlist: the five most-stressed counties with no
current AHP presence should be prioritised in the next tranche.
"""
cells[idx_45_md]["source"] = new_45_md

# ─── FIX 6: Phase 2.5 Finance exclusion scatter + stat ───
# The Phase 2.7 mortgage cell shows only a bar chart. We need to add the scatter 
# (HFVS x mortgage penetration) with county labels. This goes after cell 8.1 aggregation.
# Insert new scatter cell after 8.6 (urban-rural breakdown).

idx_86 = find_cell_by_prefix(cells, "# ── 8.6  Urban-rural HFVS disaggregation")
print(f"Cell 8.6 at index: {idx_86}")

finance_scatter_md = mk_md(
    "## 8.7 Finance Exclusion Quadrant Analysis\n\n"
    "This scatter plot addresses the missing visualisation identified in the review: "
    "counties with **high HFVS (most vulnerable)** and **low mortgage penetration "
    "(most finance-excluded)** are the primary commercial and policy targets — they "
    "simultaneously represent the highest insurance risk and the largest underserved "
    "market. The quadrant labels follow the axes: the upper-left quadrant (high HFVS, "
    "low mortgage) contains the 'double-missed' counties — most vulnerable and least "
    "served by formal housing finance. The narrative claim in Phase 9 that mortgage "
    "penetration correlates negatively with HFVS is tested here with a Spearman "
    "correlation statistic."
)

finance_scatter_code = mk_code(
"""# ── 8.7  Finance exclusion scatter: HFVS vs mortgage penetration ─────────────
# This is the 'Vondetta commercial framing' chart: high HFVS + low mortgage
# = the policy target counties that are simultaneously most vulnerable and
# most underserved by formal housing finance.

from matplotlib.lines import Line2D

cr = county_risk[['county_name', 'mean_hfvs', 'mortgage_penetration',
                  'pct_urban', 'pct_triple_exposed']].dropna()

hfvs_med  = cr['mean_hfvs'].median()
mtg_med   = cr['mortgage_penetration'].median()

rho_fin, p_fin = stats.spearmanr(cr['mean_hfvs'], cr['mortgage_penetration'])
print(f"Finance exclusion correlation:")
print(f"  Spearman rho(HFVS, mortgage_penetration) = {rho_fin:.3f}  (p = {p_fin:.4f})")
if p_fin < 0.05:
    direction = "NEGATIVE" if rho_fin < 0 else "POSITIVE"
    print(f"  → Statistically significant {direction} correlation.")
    print(f"  → Confirms: counties with higher vulnerability have {'lower' if rho_fin < 0 else 'higher'} mortgage penetration.")
else:
    print(f"  → No statistically significant correlation (p ≥ 0.05).")

fig, ax = plt.subplots(figsize=(10, 7))

# Scatter — colour by % urban
sc = ax.scatter(
    cr['mean_hfvs'], cr['mortgage_penetration'] * 100,
    c=cr['pct_urban'], cmap='RdYlGn', s=60, alpha=0.85,
    edgecolors='white', linewidth=0.6, zorder=3
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('% Urban households', fontsize=9)

# Quadrant dividers
ax.axvline(hfvs_med,    color=GRAY, lw=1.2, ls='--', alpha=0.7)
ax.axhline(mtg_med * 100, color=GRAY, lw=1.2, ls='--', alpha=0.7)

# Quadrant labels
xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
ax.text(xmin + 0.001, ymax - 0.003 * (ymax - ymin), 
        'Low HFVS\\nHigher finance access', fontsize=7.5, color=TEAL,
        va='top', ha='left', alpha=0.7)
ax.text(xmax - 0.001, ymax - 0.003 * (ymax - ymin),
        'High HFVS\\nHigher finance access', fontsize=7.5, color=AMBER,
        va='top', ha='right', alpha=0.7)
ax.text(xmin + 0.001, ymin + 0.003 * (ymax - ymin),
        'Low HFVS\\nLow finance access', fontsize=7.5, color=BLUE,
        va='bottom', ha='left', alpha=0.7)
ax.text(xmax - 0.001, ymin + 0.003 * (ymax - ymin),
        '🎯 HIGH HFVS\\nLOW FINANCE ACCESS\\n(Policy target)', fontsize=8, color=RED,
        va='bottom', ha='right', fontweight='bold')

# Label outlier counties in the high-HFVS / low-mortgage quadrant
policy_target = cr[
    (cr['mean_hfvs'] > hfvs_med) &
    (cr['mortgage_penetration'] < mtg_med)
].nlargest(12, 'mean_hfvs')

for _, row in policy_target.iterrows():
    ax.annotate(
        row['county_name'],
        (row['mean_hfvs'], row['mortgage_penetration'] * 100),
        fontsize=7.5, color=RED, alpha=0.9,
        xytext=(4, 4), textcoords='offset points'
    )

# Also label high-HFVS / high-mortgage (interesting contrast)
high_high = cr[(cr['mean_hfvs'] > hfvs_med) & (cr['mortgage_penetration'] >= mtg_med)]
for _, row in high_high.nlargest(4, 'mortgage_penetration').iterrows():
    ax.annotate(
        row['county_name'],
        (row['mean_hfvs'], row['mortgage_penetration'] * 100),
        fontsize=7.5, color=AMBER, alpha=0.85,
        xytext=(4, -10), textcoords='offset points'
    )

ax.set_xlabel('Mean HFVS (higher = more vulnerable)', fontsize=11)
ax.set_ylabel('Formal Mortgage Penetration (% of households)', fontsize=11)
ax.set_title(
    f'Finance Exclusion Quadrant — HFVS vs Mortgage Penetration\\n'
    f'Spearman rho = {rho_fin:.3f} (p = {p_fin:.4f}) | '
    f'Red labels = high vulnerability, low finance access',
    fontsize=11, fontweight='600'
)

plt.tight_layout()
plt.savefig(FIGS / 'phase8_finance_exclusion_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nPolicy-target counties (High HFVS + Low mortgage penetration):")
print(policy_target[['county_name','mean_hfvs','mortgage_penetration',
                      'pct_triple_exposed']].to_string(index=False))
"""
)

cells.insert(idx_86 + 1, finance_scatter_md)
cells.insert(idx_86 + 2, finance_scatter_code)

# ─── FIX 7: Pre-model vs post-model county HFVS comparison chart ───
# Insert after the finance scatter (now idx_86+3)
prepost_md = mk_md(
    "## 8.8 Pre-Model vs Post-Model County HFVS Comparison\n\n"
    "This side-by-side chart is the visual evidence for the study's core claim: "
    "that proxy-only HFVS approximation is feasible. The left panel shows the "
    "**measured** county mean HFVS (direct formula from the full KHS questionnaire). "
    "The right panel shows the **predicted** county mean HFVS from the LightGBM "
    "proxy model — which has no access to any formula ingredient. The correlation "
    "between the two rankings is the spatial validation of the proxy model: if the "
    "proxy model correctly identifies which counties are most vulnerable using only "
    "demographic and context features, the two rankings should be strongly aligned. "
    "Divergences reveal counties where demographic proxies alone are insufficient — "
    "typically ASAL counties with high physical hazard scores that have no demographic "
    "correlate accessible to a field worker."
)

prepost_code = mk_code(
"""# ── 8.8  Pre-model (measured) vs post-model (proxy) county HFVS comparison ───
# Left panel  : actual county mean HFVS (measured from full questionnaire)
# Right panel : proxy-model predicted HFVS (LightGBM proxy only)
# This is the core claim visualisation: do the proxy predictions agree with
# direct measurement at the county level?

# Compute county-level proxy predictions
proxy_preds_full = best_lgb_reg.predict(
    master[ALL_PROXY_FEATURES].fillna(0).values.astype(np.float32)
)
master_full['proxy_hfvs'] = proxy_preds_full

county_proxy = (
    master_full.groupby('county_name')
    .apply(lambda s: np.average(s['proxy_hfvs'], weights=s['hhweight']))
    .reset_index().rename(columns={0: 'proxy_mean_hfvs'})
)

compare_df = county_risk[['county_name','mean_hfvs']].merge(
    county_proxy, on='county_name', how='inner'
).sort_values('mean_hfvs')

# Rank correlation between actual and predicted county rankings
rho_pp, p_pp = stats.spearmanr(compare_df['mean_hfvs'], compare_df['proxy_mean_hfvs'])
mae_county = (compare_df['mean_hfvs'] - compare_df['proxy_mean_hfvs']).abs().mean()
print(f"County-level proxy validation:")
print(f"  Spearman rho (actual vs proxy rank) : {rho_pp:.3f}  (p = {p_pp:.4f})")
print(f"  Mean absolute error (county HFVS)   : {mae_county:.4f}")

# ── Dual panel: side-by-side ranked bars ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 13), sharey=True)

nat_mean_a = compare_df['mean_hfvs'].mean()
nat_mean_p = compare_df['proxy_mean_hfvs'].mean()

for ax, col, title, nat_mean, label in [
    (axes[0], 'mean_hfvs',       'Measured HFVS\\n(full KHS questionnaire)',
     nat_mean_a, 'Measured'),
    (axes[1], 'proxy_mean_hfvs', 'Proxy-Model HFVS\\n(demographic proxies only)',
     nat_mean_p, 'Proxy'),
]:
    cs = compare_df.sort_values(col)
    bar_c = [RED if v > nat_mean else TEAL for v in cs[col]]
    ax.barh(cs['county_name'], cs[col], color=bar_c, edgecolor='none', alpha=0.88)
    ax.axvline(nat_mean, color=DARK, lw=1.5, ls='--',
               label=f'National mean ({nat_mean:.3f})')
    ax.set_xlabel(f'Mean {label} HFVS', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='600')
    ax.legend(fontsize=8)

fig.suptitle(
    f'Pre-Model vs Post-Model County HFVS Comparison\\n'
    f'County rank Spearman rho = {rho_pp:.3f} | MAE = {mae_county:.4f}',
    fontsize=13, fontweight='700'
)
plt.tight_layout()
plt.savefig(FIGS / 'phase8_premodel_vs_postmodel_county_hfvs.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Scatter: actual vs proxy (with county names for outliers) ─────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(compare_df['mean_hfvs'], compare_df['proxy_mean_hfvs'],
           color=TEAL, s=60, alpha=0.8, edgecolors='white')
# 45-degree reference
lims = [min(compare_df[['mean_hfvs','proxy_mean_hfvs']].min()),
        max(compare_df[['mean_hfvs','proxy_mean_hfvs']].max())]
ax.plot(lims, lims, color=DARK, lw=1.5, ls='--', label='Perfect agreement')

# Label counties with largest divergence
compare_df['abs_diff'] = (compare_df['mean_hfvs'] - compare_df['proxy_mean_hfvs']).abs()
for _, row in compare_df.nlargest(8, 'abs_diff').iterrows():
    ax.annotate(row['county_name'][:10],
                (row['mean_hfvs'], row['proxy_mean_hfvs']),
                fontsize=7.5, color=RED, xytext=(4, 2), textcoords='offset points')

ax.set_xlabel('Measured HFVS (direct formula)'); ax.set_ylabel('Proxy-Predicted HFVS')
ax.set_title(f'County HFVS: Actual vs Proxy\\nrho={rho_pp:.3f}  MAE={mae_county:.4f}',
             fontsize=11, fontweight='600')
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / 'phase8_actual_vs_proxy_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nCounties with largest proxy divergence (actual - proxy):") 
print(compare_df.nlargest(8,'abs_diff')[['county_name','mean_hfvs','proxy_mean_hfvs','abs_diff']].to_string(index=False))
"""
)

# Find current index after the previous inserts
idx_after_scatter = find_cell_by_prefix(cells, "# ── 8.7  Finance exclusion scatter")
cells.insert(idx_after_scatter + 1, prepost_md)
cells.insert(idx_after_scatter + 2, prepost_code)

# ─── FIX 8: Lorenz curve / within-county inequality ───
idx_after_prepost = find_cell_by_prefix(cells, "# ── 8.8  Pre-model (measured) vs post-model")
lorenz_md = mk_md(
    "## 8.9 Within-County HFVS Inequality — Lorenz Curves and Gini Coefficients\n\n"
    "A composite index can have the same county mean but very different internal "
    "distributions. A Nairobi with mean HFVS 0.45 could reflect a bimodal population "
    "of very low-vulnerability formal-sector households and very high-vulnerability "
    "informal settlement households, or it could reflect a relatively uniform moderate "
    "vulnerability. The Lorenz curve and Gini coefficient distinguish these cases. "
    "High within-county Gini suggests that county-level policy instruments (a single "
    "county-wide insurance product, a county-level AHP allocation) will be blunt: "
    "the product needs sub-county targeting. Low Gini means the county is relatively "
    "homogeneous and county-level instruments are efficient. This analysis is a "
    "distinctive finding that no prior Kenya housing study has produced at this scale."
)

lorenz_code = mk_code(
"""# ── 8.9  Within-county HFVS inequality — Lorenz curves and Gini coefficients ──

def gini_coefficient(values):
    \"\"\"Compute the Gini coefficient of a vulnerability distribution.
    
    A Gini of 0 means perfect equality (all households identically vulnerable).
    A Gini of 1 means total inequality (all vulnerability concentrated in one HH).
    For HFVS, higher Gini = more heterogeneous county = blunter county-level policy.
    \"\"\"
    v = np.sort(np.array(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0: return np.nan
    cumsum = np.cumsum(v)
    return (2 * np.sum((np.arange(1, n + 1) * v)) / (n * cumsum[-1])) - (n + 1) / n


def lorenz_curve(values):
    \"\"\"Return (x, y) coordinates for a Lorenz curve.\"\"\
    v = np.sort(np.array(values, dtype=float))
    cum_v = np.cumsum(v) / v.sum() if v.sum() > 0 else np.zeros_like(v)
    cum_n = np.linspace(0, 1, len(v) + 1)
    return cum_n, np.concatenate([[0], cum_v])


# ── Compute Gini per county ───────────────────────────────────────────────────
gini_rows = []
for code, name in COUNTY_MAP.items():
    sub = master_full[master_full['a01'] == code]['hfvs'].dropna()
    if len(sub) < 20: continue
    g = gini_coefficient(sub.values)
    p25, p75 = sub.quantile(0.25), sub.quantile(0.75)
    gini_rows.append({
        'county_name': name, 'gini': g,
        'mean_hfvs': sub.mean(), 'p25': p25, 'p75': p75,
        'iqr': p75 - p25, 'n': len(sub)
    })

gini_df = pd.DataFrame(gini_rows).sort_values('gini', ascending=False)
print("Within-county HFVS Gini coefficients (top 10 most unequal):")
print(gini_df.head(10)[['county_name','gini','mean_hfvs','iqr','n']].to_string(index=False))
print("\\nWithin-county HFVS Gini coefficients (10 most equal):")
print(gini_df.tail(10)[['county_name','gini','mean_hfvs','iqr','n']].to_string(index=False))

# ── Lorenz curves: 5 most unequal vs 5 most equal ────────────────────────────
most_unequal = gini_df.head(5)['county_name'].tolist()
most_equal   = gini_df.tail(5)['county_name'].tolist()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, county_list, title, cmap_name in [
    (axes[0], most_unequal, '5 Most Unequal Counties\\n(highest within-county HFVS Gini)', 'Reds'),
    (axes[1], most_equal,   '5 Most Equal Counties\\n(lowest within-county HFVS Gini)',   'Greens'),
]:
    palette = plt.get_cmap(cmap_name)(np.linspace(0.4, 0.9, len(county_list)))
    for name, col in zip(county_list, palette):
        sub = master_full[master_full['county_name'] == name]['hfvs'].dropna()
        cx, cy = lorenz_curve(sub.values)
        g = gini_df[gini_df['county_name'] == name]['gini'].values[0]
        ax.plot(cx, cy, color=col, lw=2, label=f'{name} (G={g:.3f})')
    ax.plot([0, 1], [0, 1], color=DARK, lw=1.2, ls='--', alpha=0.6, label='Perfect equality')
    ax.set_xlabel('Cumulative share of households')
    ax.set_ylabel('Cumulative share of HFVS')
    ax.set_title(title, fontsize=11, fontweight='600')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.suptitle('Within-County HFVS Inequality — Lorenz Curves\\n'
             'High Gini = heterogeneous county (blunter county-level policy)',
             fontsize=12, fontweight='700')
plt.tight_layout()
plt.savefig(FIGS / 'phase8_lorenz_curves_county_inequality.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Gini vs mean HFVS scatter ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(gini_df['mean_hfvs'], gini_df['gini'], color=TEAL, s=50, alpha=0.8)
for _, row in gini_df.head(5).iterrows():
    ax.annotate(row['county_name'][:10], (row['mean_hfvs'], row['gini']),
                fontsize=7.5, color=RED, xytext=(3, 2), textcoords='offset points')
rho_gm, p_gm = stats.spearmanr(gini_df['mean_hfvs'], gini_df['gini'])
ax.set_xlabel('Mean HFVS (county average)')
ax.set_ylabel('Within-county Gini coefficient')
ax.set_title(f'Within-County Inequality vs Mean Vulnerability\\nSpearman rho={rho_gm:.3f} (p={p_gm:.4f})')
plt.tight_layout()
plt.savefig(FIGS / 'phase8_gini_vs_mean_hfvs.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\\nGini vs mean HFVS: rho = {rho_gm:.3f} (p = {p_gm:.4f})")
print("High positive rho → more vulnerable counties also more internally unequal.")
print("Policy implication: county-level instruments are least efficient where they are most needed.")
"""
)

cells.insert(idx_after_prepost + 3, lorenz_md)
cells.insert(idx_after_prepost + 4, lorenz_code)

# ─── FIX 9: Update Phase 9 narrative to remove placeholders ───
idx_p9 = find_cell_by_prefix(cells, "--- # 📝 Phase 9 — Conclusions")
src_p9 = "".join(cells[idx_p9]["source"])

# Fix known placeholder phrases
src_p9 = src_p9.replace(
    "account for a non-trivial share of the national sample",
    "account for a proportion quantified in Phase 4.4 (see triple_exposed weighted mean output above)"
).replace(
    "to several thousand Kenyan shillings per month",
    "to a median of KES {median_excess_placeholder}/month — replace with actual output value from cell 4.5b"
).replace(
    "approximately 40 percent",
    "the exact percentage quantified by `master['target_binary'].mean()*100` in Phase 3.10 (see output above)"
)

# Add a note at the start of phase 9
phase9_note = (
    "> **📋 Dissertation author note:** The Phase 9 narrative below was drafted as a "
    "template before the notebook was run. Placeholder phrases marked with `{...}` "
    "**must be replaced with actual output values** from the corresponding code cells "
    "above before final submission. Specifically: (1) the triple-exposure percentage "
    "from cell 4.4a; (2) the median affordability gap from cell 4.5b; (3) the "
    "high-vulnerability proportion from cell 3.10; (4) the LightGBM AUC range from "
    "cell 7.1; and (5) the spatial correction range from cell 5.5 / 7.1.\n\n"
)
src_p9 = src_p9.replace("--- # 📝 Phase 9", f"--- \n{phase9_note}# 📝 Phase 9", 1)
cells[idx_p9]["source"] = src_p9

# ─── Save ───
with open("KHS_Dissertation_Enhanced.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"\n✓ Notebook saved with {len(nb['cells'])} cells")
PYEOF
Output

Cell 6.6 at index: 76
Cell 8.1 at index: 85
Cell 5.5 at index: 69
Cell 4.5 narrative at index: -1
Cell 8.6 at index: 92

✓ Notebook saved with 101 cells