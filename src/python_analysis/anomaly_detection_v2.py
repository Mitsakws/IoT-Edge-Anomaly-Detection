# Thesis Project: Anomaly Detection on IoT Temperature Data
# Author: Dimitris Kostoulas
# Description: Resampling, Feature Engineering, Train/Test Split and Isolation Forest evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Plotting Palette
BG        = '#0F1117'
PANEL     = '#1A1D2E'
INDOOR    = '#00D4FF'
OUTDOOR   = '#FF9500'
ANOMALY   = '#FF3B5C'
GRID_C    = '#2A2D3E'
TEXT      = '#E0E0E0'
GREEN     = '#A8FF78'
PURPLE    = '#C9B1FF'
YELLOW    = '#FFD166'
ORANGE    = '#F4845F'

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.7)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)

# 1. LOAD DATA
print("Starting Anomaly Detection Pipeline...")
df_raw = pd.read_csv('C:\\Users\\giaka\\Downloads\\feeds.csv')
df_raw['created_at'] = pd.to_datetime(df_raw['created_at'], utc=True)
df_raw = df_raw[['created_at','field1','field2']].dropna()
df_raw = df_raw.sort_values('created_at').reset_index(drop=True)
df_raw.columns = ['timestamp','temp_indoor','temp_outdoor']
print(f"Loaded {len(df_raw)} raw records (~1 min intervals)")

# 2. RESAMPLE TO 10 MINUTES
df = df_raw.set_index('timestamp').resample('10min').mean().dropna().reset_index()
print(f"Resampled to {len(df)} records (10-min intervals)")

# 3. FEATURE ENGINEERING
print("Computing dynamic features...")
df['roc_10min']       = df['temp_indoor'].diff(1)
df['roc_30min']       = df['temp_indoor'].diff(3)
df['rolling_std_30']  = df['temp_indoor'].rolling(3, min_periods=2).std()
thermal_gap           = df['temp_indoor'] - df['temp_outdoor']
df['delta_thermal_gap'] = thermal_gap.diff(1)
df['acceleration']    = df['roc_10min'].diff(1)

feature_cols = ['roc_10min','roc_30min','rolling_std_30','delta_thermal_gap','acceleration']
df_m = df.dropna(subset=feature_cols).copy()

# 4. TRAIN / TEST SPLIT ( 70/30)
print("Splitting dataset into Train (70%) and Test (30%) phases...")
split_idx = int(len(df_m) * 0.7)
df_train = df_m.iloc[:split_idx].copy()
df_test  = df_m.iloc[split_idx:].copy()
split_date = df_test['timestamp'].iloc[0]

print(f"Train Set: {len(df_train)} records (Ends at {split_date})")
print(f"Test Set : {len(df_test)} records (Starts at {split_date})")

# 5. SCALING (Fit ONLY)
print("Applying scalers...")
rs = RobustScaler()
ss = StandardScaler()

# Fit only on Train, transform both Train and Test
X_train_robust = rs.fit_transform(df_train[feature_cols])
X_test_robust  = rs.transform(df_test[feature_cols])
X_robust = np.vstack((X_train_robust, X_test_robust)) # Combined for overall scoring/plotting

X_train_standard = ss.fit_transform(df_train[feature_cols])
X_test_standard  = ss.transform(df_test[feature_cols])
X_standard = np.vstack((X_train_standard, X_test_standard))

spikes_robust   = (np.abs(X_robust)   > 3).sum(axis=1)
spikes_standard = (np.abs(X_standard) > 3).sum(axis=1)
print(f"Spikes preserved (RobustScaler): {(spikes_robust>0).sum()}")
print(f"Spikes preserved (StandardScaler): {(spikes_standard>0).sum()}")

# 6. ISOLATION FOREST (Train ONLY  Train Set)
print("Training Isolation Forest on Train set...")
iforest = IsolationForest(
    n_estimators   = 200,
    contamination  = 0.04,
    max_samples    = 'auto',
    random_state   = 42,
    n_jobs         = -1
)
iforest.fit(X_train_robust) # ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΟ ΣΤΟ TRAIN SET

# Predict on the entire dataset (Train + Test)
print("Evaluating model on unseen Test data...")
df_m['anomaly_label'] = iforest.predict(X_robust)
df_m['anomaly_score'] = iforest.score_samples(X_robust)

n_anom    = (df_m['anomaly_label'] == -1).sum()
pct_anom  = 100 * n_anom / len(df_m)
threshold = df_m.loc[df_m['anomaly_label']==-1, 'anomaly_score'].max()

print(f"Detected {n_anom} anomalies across entire dataset.")
print(f"Score threshold: {threshold:.4f}")

anomalies = df_m[df_m['anomaly_label']==-1].sort_values('anomaly_score')
mask = df_m['anomaly_label'] == -1

# 7. GENERATE PLOTS
print("Generating Figure 1 (Main Time-Series)...")
fig1, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios':[3, 1.2, 1.2]})
fig1.patch.set_facecolor(BG)

# Panel 1: Temperatures
ax = axes[0]; style_ax(ax)
ax.fill_between(df_m['timestamp'], df_m['temp_indoor'], df_m['temp_outdoor'], alpha=0.07, color=INDOOR)
ax.plot(df_m['timestamp'], df_m['temp_outdoor'], color=OUTDOOR, lw=1.3, alpha=0.85, label='Outdoor Temp', zorder=2)
ax.plot(df_m['timestamp'], df_m['temp_indoor'], color=INDOOR, lw=1.8, alpha=0.95, label='Indoor Temp', zorder=3)
ax.scatter(df_m.loc[mask,'timestamp'], df_m.loc[mask,'temp_indoor'], color=ANOMALY, s=70, zorder=5, label=f'Anomaly (n={n_anom})', edgecolors='white', linewidths=0.6)

# TRAIN / TEST SPLIT VISUALIZATION
ax.axvline(split_date, color=YELLOW, lw=2, ls='--', alpha=0.9, zorder=10)
ax.text(split_date, df_m['temp_indoor'].max(), '  TEST SET (Unseen) ➡️', color=YELLOW, va='top', ha='left', fontsize=10, fontweight='bold')
ax.text(split_date, df_m['temp_indoor'].max(), '⬅️ TRAIN SET  ', color=YELLOW, va='top', ha='right', fontsize=10, fontweight='bold')

for t in anomalies.head(5)['timestamp']:
    ax.axvline(x=t, color=ANOMALY, alpha=0.12, lw=1, ls='--')
ax.set_ylabel('Temperature (C)', color=TEXT, fontsize=11)
ax.set_title('Temperature Anomaly Detection — Isolation Forest | Train/Test Split', color=TEXT, fontsize=13, fontweight='bold', pad=10)
ax.legend(loc='upper right', framealpha=0.3, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT, fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
plt.setp(ax.get_xticklabels(), visible=False)

# Panel 2: RoC
ax2 = axes[1]; style_ax(ax2)
ax2.plot(df_m['timestamp'], df_m['roc_10min'], color=GREEN, lw=0.9, alpha=0.75, label='RoC per 10 min')
ax2.fill_between(df_m['timestamp'], df_m['roc_10min'], 0, where=(df_m['roc_10min']<0), color='#FF6B6B', alpha=0.3, label='Drop')
ax2.fill_between(df_m['timestamp'], df_m['roc_10min'], 0, where=(df_m['roc_10min']>=0), color=GREEN, alpha=0.15, label='Rise')
ax2.scatter(df_m.loc[mask,'timestamp'], df_m.loc[mask,'roc_10min'], color=ANOMALY, s=40, zorder=5, alpha=0.9)
ax2.axhline(0, color=TEXT, lw=0.5, alpha=0.5)
roc_std = df_m['roc_10min'].std()
ax2.axhline(-2*roc_std, color=ANOMALY, lw=0.9, ls=':', alpha=0.55, label=f'+-2sigma ({2*roc_std:.3f})')
ax2.axhline( 2*roc_std, color=ANOMALY, lw=0.9, ls=':', alpha=0.55)
ax2.axvline(split_date, color=YELLOW, lw=1.5, ls='--', alpha=0.7)
ax2.set_ylabel('C / 10 min', color=TEXT, fontsize=10)
ax2.set_title('Indoor Temperature Rate of Change (10-min step)', color=TEXT, fontsize=10, pad=5)
ax2.legend(loc='upper right', framealpha=0.2, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT, fontsize=8, ncol=4)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
plt.setp(ax2.get_xticklabels(), visible=False)

# Panel 3: Score
ax3 = axes[2]; style_ax(ax3)
ax3.fill_between(df_m['timestamp'], df_m['anomaly_score'], df_m['anomaly_score'].max(), color='#4A4E69', alpha=0.3)
ax3.plot(df_m['timestamp'], df_m['anomaly_score'], color=PURPLE, lw=0.9, alpha=0.85, label='Anomaly Score')
ax3.fill_between(df_m['timestamp'], df_m['anomaly_score'], threshold, where=(df_m['anomaly_score']<=threshold), color=ANOMALY, alpha=0.28, label='Anomaly zone')
ax3.axhline(threshold, color=ANOMALY, lw=1.1, ls='--', alpha=0.8, label=f'Threshold = {threshold:.4f}')
ax3.scatter(df_m.loc[mask,'timestamp'], df_m.loc[mask,'anomaly_score'], color=ANOMALY, s=25, zorder=5, alpha=0.95)
ax3.axvline(split_date, color=YELLOW, lw=1.5, ls='--', alpha=0.7)
ax3.set_ylabel('Score', color=TEXT, fontsize=10)
ax3.set_title('Anomaly Score', color=TEXT, fontsize=10, pad=5)
ax3.legend(loc='lower right', framealpha=0.2, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT, fontsize=8)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.setp(ax3.get_xticklabels(), rotation=28, ha='right', color=TEXT)

plt.tight_layout(pad=1.5); fig1.subplots_adjust(hspace=0.09)
fig1.savefig('anomaly_detection_v2_main.png', dpi=150, bbox_inches='tight', facecolor=BG)

print("Generating Figure 2 (Scaler Comparison)...")
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
fig2.patch.set_facecolor(BG)
fig2.suptitle('RobustScaler vs StandardScaler', color=TEXT, fontsize=13, fontweight='bold', y=0.98)

feat_idx    = [0, 3, 4]
feat_names  = ['RoC 10 min (C/10min)', 'Delta Thermal Gap (C/10min)', 'Acceleration (C/10min^2)']
colors_r    = [INDOOR, GREEN, PURPLE]
colors_s    = [ORANGE, YELLOW, '#FF6B6B']

for col, (fi, fn, cr, cs) in enumerate(zip(feat_idx, feat_names, colors_r, colors_s)):
    xr = X_robust[:, fi]
    xs = X_standard[:, fi]
    bins = np.linspace(min(xr.min(), xs.min()), max(xr.max(), xs.max()), 60)

    ax = axes2[0, col]; style_ax(ax)
    ax.hist(xr, bins=bins, alpha=0.6, color=cr, label='RobustScaler', density=True)
    ax.hist(xs, bins=bins, alpha=0.5, color=cs, label='StandardScaler', density=True, linestyle='--')
    ax.axvline(-3, color=ANOMALY, lw=1.2, ls='--', alpha=0.8, label='|z|=3 boundary')
    ax.axvline( 3, color=ANOMALY, lw=1.2, ls='--', alpha=0.8)
    ax.set_title(fn, color=TEXT, fontsize=9, pad=4)
    ax.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C, framealpha=0.4)
    ax.set_ylabel('Density', color=TEXT, fontsize=8)

    ax2_ = axes2[1, col]; style_ax(ax2_)
    ax2_.plot(df_m['timestamp'].values, xr, color=cr, lw=0.8, alpha=0.8, label='RobustScaler')
    ax2_.plot(df_m['timestamp'].values, xs, color=cs, lw=0.8, alpha=0.6, linestyle='--', label='StandardScaler')
    ax2_.scatter(df_m.loc[mask,'timestamp'].values, xr[mask.values], color=ANOMALY, s=30, zorder=5, label='Anomaly')
    ax2_.axvline(split_date, color=YELLOW, lw=1, ls='--', alpha=0.5)
    ax2_.axhline( 3, color=ANOMALY, lw=0.9, ls=':', alpha=0.6)
    ax2_.axhline(-3, color=ANOMALY, lw=0.9, ls=':', alpha=0.6)
    ax2_.axhline(0,  color=TEXT, lw=0.5, alpha=0.4)
    ax2_.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C, framealpha=0.4)
    ax2_.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.setp(ax2_.get_xticklabels(), rotation=25, ha='right', color=TEXT, fontsize=7)

summary = (
    "RobustScaler formula:  z = (x - median) / IQR\n"
    "StandardScaler formula: z = (x - mean) / std\n\n"
    "Key difference: RobustScaler uses MEDIAN and IQR (Q75-Q25),\n"
    "which are not affected by extreme values (spikes).\n"
    "Result: anomalous spikes are scaled to LARGER absolute values,\n"
    "making them EASIER for Isolation Forest to isolate."
)
fig2.text(0.5, 0.01, summary, ha='center', va='bottom', color=YELLOW, fontsize=8.5, bbox=dict(fc=PANEL, ec=GRID_C, alpha=0.85, boxstyle='round,pad=0.6'), linespacing=1.6)

plt.tight_layout(rect=[0, 0.10, 1, 0.97])
fig2.savefig('anomaly_detection_v2_scaler.png', dpi=150, bbox_inches='tight', facecolor=BG)

print("Generating Figure 3 (Math Foundation)...")
fig3 = plt.figure(figsize=(16, 10))
fig3.patch.set_facecolor(BG)
fig3.suptitle('Isolation Forest & RobustScaler — Mathematical Foundation', color=TEXT, fontsize=13, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.45, wspace=0.35)

ax_path = fig3.add_subplot(gs[0, 0]); style_ax(ax_path)
scores_normal  = df_m.loc[df_m['anomaly_label']== 1, 'anomaly_score'].values
scores_anomaly = df_m.loc[df_m['anomaly_label']==-1, 'anomaly_score'].values
bins_path = np.linspace(df_m['anomaly_score'].min(), df_m['anomaly_score'].max(), 45)
ax_path.hist(scores_normal, bins=bins_path, color=INDOOR, alpha=0.7, density=True, label='Normal')
ax_path.hist(scores_anomaly, bins=bins_path, color=ANOMALY, alpha=0.8, density=True, label='Anomaly')
ax_path.axvline(threshold, color=YELLOW, lw=1.5, ls='--', label=f'Threshold={threshold:.3f}')
ax_path.set_title('Anomaly Score Distribution', color=TEXT, fontsize=9)
ax_path.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C, framealpha=0.5)

ax_feat = fig3.add_subplot(gs[0, 1]); style_ax(ax_feat)
ax_feat.scatter(np.abs(X_robust[:, 0]), df_m['anomaly_score'], c=df_m['anomaly_label'].map({1: INDOOR, -1: ANOMALY}), s=20, alpha=0.6)
ax_feat.axhline(threshold, color=YELLOW, lw=1.2, ls='--', label=f'Threshold={threshold:.3f}')
ax_feat.set_xlabel('|RoC 10min| (RobustScaled)', color=TEXT, fontsize=8)
ax_feat.set_ylabel('Anomaly Score', color=TEXT, fontsize=8)
ax_feat.set_title('Score vs Feature Magnitude', color=TEXT, fontsize=9)
ax_feat.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C, framealpha=0.5)

ax_rob = fig3.add_subplot(gs[0, 2]); style_ax(ax_rob)
raw_feat = df_train['roc_10min'].values # MEDIAN IQR based on train set
med, q25, q75 = np.median(raw_feat), np.percentile(raw_feat, 25), np.percentile(raw_feat, 75)
ax_rob.hist(raw_feat, bins=50, color=INDOOR, alpha=0.65, density=True, label='Raw RoC 10min')
ax_rob.axvline(med, color=GREEN, lw=2, label=f'Median = {med:.4f}')
ax_rob.fill_betweenx([0, ax_rob.get_ylim()[1] if ax_rob.get_ylim()[1]>0 else 5], q25, q75, color=PURPLE, alpha=0.12, label=f'IQR = {q75-q25:.4f}')
ax_rob.set_title('RobustScaler Parameters (Calculated on Train Set)', color=TEXT, fontsize=9)
ax_rob.legend(fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C, framealpha=0.5)

ax_if = fig3.add_subplot(gs[1, 0]); style_ax(ax_if)
np.random.seed(42)
n_sim = 300
normal_pts = np.random.randn(n_sim) * 0.8
outlier_pts = np.array([-4.5, -3.8, 4.2, 3.9, -5.1])
path_normal = np.random.normal(12, 2, n_sim)
path_outlier = np.random.normal(4, 1, len(outlier_pts))
ax_if.scatter(normal_pts, path_normal, color=INDOOR, s=15, alpha=0.5, label='Normal')
ax_if.scatter(outlier_pts, path_outlier, color=ANOMALY, s=60, alpha=0.9, edgecolors='white', lw=0.6, label='Anomaly')
ax_if.axhline(6, color=YELLOW, lw=1.5, ls='--', label='Short path threshold')
ax_if.set_xlabel('Feature value', color=TEXT, fontsize=8)
ax_if.set_ylabel('Avg path length h(x)', color=TEXT, fontsize=8)
ax_if.set_title('Isolation Forest Concept', color=TEXT, fontsize=9)
ax_if.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C, framealpha=0.5)

ax_imp = fig3.add_subplot(gs[1, 1]); style_ax(ax_imp)
importance = [np.abs(X_robust[mask.values, i]).mean() / (np.abs(X_robust[~mask.values, i]).mean() or 1) for i in range(len(feature_cols))]
feat_short = ['RoC\n10min','RoC\n30min','Roll.\nStd30','Delta\nGap','Accel.']
bars = ax_imp.bar(feat_short, importance, color=[INDOOR, GREEN, PURPLE, YELLOW, ORANGE], alpha=0.85, edgecolor=GRID_C, linewidth=0.7)
ax_imp.axhline(1.0, color=TEXT, lw=1, ls='--', alpha=0.5, label='Baseline')
ax_imp.set_ylabel('Anomaly / Normal ratio', color=TEXT, fontsize=8)
ax_imp.set_title('Feature Signal Strength', color=TEXT, fontsize=9)
for b, v in zip(bars, importance):
    ax_imp.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'{v:.2f}x', ha='center', va='bottom', color=TEXT, fontsize=8)

ax_ee = fig3.add_subplot(gs[1, 2]); style_ax(ax_ee)
intervals = [1, 5, 10, 15, 30, 60]
tx_per_day = [24*60//i for i in intervals]
colors_ee = [ANOMALY if i==1 else (YELLOW if i==10 else INDOOR) for i in intervals]
bars_ee = ax_ee.bar([f'{i} min' for i in intervals], tx_per_day, color=colors_ee, alpha=0.85, edgecolor=GRID_C, lw=0.7)
ax_ee.set_ylabel('Transmissions / day', color=TEXT, fontsize=8)
ax_ee.set_title('Energy Efficiency', color=TEXT, fontsize=9)
for b, v in zip(bars_ee, tx_per_day):
    ax_ee.text(b.get_x() + b.get_width()/2, b.get_height() + 3, str(v), ha='center', va='bottom', color=TEXT, fontsize=8)

fig3.savefig('anomaly_detection_v2_math.png', dpi=150, bbox_inches='tight', facecolor=BG)

print("Generating Figure 4 (Algorithm Comparison)...")
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 6))
fig4.patch.set_facecolor(BG)
fig4.suptitle('Method Justification', color=TEXT, fontsize=13, fontweight='bold')

ax = axes4[0]; style_ax(ax)
sc = ax.scatter(df_m['timestamp'], df_m['anomaly_score'], c=df_m['anomaly_score'], cmap='RdYlGn', s=15, alpha=0.8, zorder=3)
ax.axvline(split_date, color=YELLOW, lw=2, ls='--', alpha=0.9, zorder=10)
ax.axhline(threshold, color=ANOMALY, lw=1.2, ls='--', label=f'Decision boundary = {threshold:.4f}')
ax.set_title('Anomaly Score Timeline', color=TEXT, fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
plt.setp(ax.get_xticklabels(), rotation=25, ha='right', color=TEXT, fontsize=7)

ax = axes4[1]; style_ax(ax)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
table_data = [
    ['Method',          'No Labels', 'Edge OK', 'Fast', 'Score'],
    ['Isolation Forest','    YES',    '   YES',  ' YES', ' YES'],
    ['LOF',             '    YES',    '    NO',  '  NO', ' YES'],
    ['DBSCAN',          '    YES',    '    OK',  ' YES', '  NO'],
    ['Autoencoder',     '    YES',    '    NO',  '  NO', ' YES'],
    ['Z-score',         '    YES',    '   YES',  ' YES', '  NO'],
]
col_x = [0.5, 3.2, 5.2, 7.0, 8.8]
row_y = [9.2, 7.8, 6.5, 5.2, 3.9, 2.6]
for j, (cx, hdr) in enumerate(zip(col_x, table_data[0])):
    ax.text(cx, row_y[0], hdr, color=YELLOW, fontsize=8.5, fontweight='bold', va='center')
for i, row in enumerate(table_data[1:]):
    if i == 0: ax.add_patch(plt.Rectangle((0, row_y[i+1]-0.55), 10, 1.1, fc=INDOOR, alpha=0.15, zorder=0))
    for j, (cx, val) in enumerate(zip(col_x, row)):
        color = GREEN if 'YES' in val else (ANOMALY if 'NO' in val else YELLOW)
        ax.text(cx, row_y[i+1], val, color=color, fontsize=8, va='center')
ax.set_title('Algorithm Comparison', color=TEXT, fontsize=9)

ax = axes4[2]; style_ax(ax)
contams = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
n_anoms_c = [(IsolationForest(n_estimators=200, contamination=c, random_state=42, n_jobs=-1).fit(X_train_robust).predict(X_robust)==-1).sum() for c in contams]
ax.plot(contams, n_anoms_c, color=INDOOR, lw=2, marker='o')
ax.axvline(0.04, color=YELLOW, lw=1.5, ls='--', label='Chosen: 0.04')
ax.set_xlabel('contamination parameter', color=TEXT, fontsize=8)
ax.set_title('Contamination Sensitivity', color=TEXT, fontsize=9)
ax.legend(fontsize=7.5, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C, framealpha=0.5)

plt.tight_layout(pad=2)
fig4.savefig('anomaly_detection_v2_why_iforest.png', dpi=150, bbox_inches='tight', facecolor=BG)

out = df_m[['timestamp','temp_indoor','temp_outdoor'] + feature_cols + ['anomaly_label','anomaly_score']].copy()
out.to_csv('anomaly_results_v2.csv', index=False)
print("Saved outputs and completed pipeline.")
