"""
Greenhouse Microclimate Analysis Script
========================================
Designed for: Semi-high-tech greenhouse thesis research (Nepal mid-hill region)
Data format:  1-minute interval weather station logger (.xlsx)
Author note:  Adapt file path and treatment labels when you have T1/T2/T3 data

Usage:
    python greenhouse_analysis.py

Outputs (saved in output/ folder):
    - daily_summary.csv          : Daily max/min/mean stats
    - hourly_diurnal.csv         : Average hourly values across all days
    - fig1_diurnal_pattern.png   : Temperature + humidity + GHI diurnal curve
    - fig2_daily_range.png       : Daily temperature range + humidity trend
    - fig3_solar_radiation.png   : Daily solar irradiance (max & mean)
    - fig4_soil_temperature.png  : Soil temperature profile across depths
    - fig5_wind_rose.png         : Wind speed and direction rose
    - fig6_correlation.png       : GHI vs temperature scatter + regression
    - full_report_table.txt      : Printable summary statistics table
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from scipy import stats

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION  ← change paths/labels here
# ─────────────────────────────────────────────
FILE_PATH   = r"C:\Users\Arbind\Documents\New folder\2024-2-1_12-20--to--2024-2-8_12-20.xlsx"  # your data file
OUTPUT_DIR  = "output"                                      # where figures are saved
SITE_LABEL  = "Outdoor Station – Chitlang (Feb 2024)"      # appears on plot titles
TREATMENT   = "Outdoor (baseline)"                          # T1 / T2 / T3 / Outdoor

# Plot style
TEMP_COLOR  = "#D85A30"   # coral-red for temperature
HUM_COLOR   = "#185FA5"   # blue for humidity
GHI_COLOR   = "#BA7517"   # amber for solar
SOIL_COLORS = ["#3B6D11", "#639922", "#97C459", "#C0DD97",
               "#E1F5EE", "#9FE1CB", "#5DCAA5"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────
print("Loading data ...")
df = pd.read_excel(FILE_PATH)
df["recorded_at"] = pd.to_datetime(df["recorded_at"])
df = df.sort_values("recorded_at").reset_index(drop=True)

df["date"]     = df["recorded_at"].dt.date
df["hour"]     = df["recorded_at"].dt.hour
df["datetime"] = df["recorded_at"]

# Day vs Night classification (Day = 6:00–17:59, Night = 18:00–5:59)
DAY_START_HOUR = 6
DAY_END_HOUR   = 18
df["period"] = np.where(
    (df["hour"] >= DAY_START_HOUR) & (df["hour"] < DAY_END_HOUR),
    "Day", "Night"
)

# GHI sensor reads nothing at night (no sunlight) — treat as 0, not missing
df["ghi_solar_irradiance"] = df["ghi_solar_irradiance"].fillna(0)

# Soil columns present in this dataset
SOIL_COLS   = [c for c in df.columns if c.startswith("soil_temp") and "r1c1" not in c and "moisture" not in c]
SOIL_LABELS = [f"Depth {i+1}" for i in range(len(SOIL_COLS))]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"  Rows: {len(df):,}  |  Date range: {df['date'].min()} → {df['date'].max()}")

# ── Helper: circular mean wind direction (vector averaging) ──
# A plain arithmetic mean is wrong for angles (e.g. mean of 350° and 10° should be 0°, not 180°)
def circular_mean_deg(angles_deg):
    angles_deg = angles_deg.dropna()
    if len(angles_deg) == 0:
        return np.nan
    rad = np.deg2rad(angles_deg)
    mean_sin = np.sin(rad).mean()
    mean_cos = np.cos(rad).mean()
    mean_angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))
    return mean_angle % 360

COMPASS_DIRS = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                'S','SSW','SW','WSW','W','WNW','NW','NNW']

def deg_to_compass(deg):
    if pd.isna(deg):
        return None
    idx = int((deg / 22.5) + 0.5) % 16
    return COMPASS_DIRS[idx]

def dominant_compass(angles_deg):
    """Most frequent 16-point compass sector (mode of binned directions)."""
    angles_deg = angles_deg.dropna()
    if len(angles_deg) == 0:
        return None
    bins = np.linspace(0, 360, 17)
    binned = pd.cut(angles_deg, bins=bins, labels=COMPASS_DIRS, right=False, include_lowest=True)
    return binned.value_counts().idxmax()

def dominant_pct(angles_deg):
    """% of readings that fall in the dominant 16-point sector — tells you how steady/consistent the wind direction was."""
    angles_deg = angles_deg.dropna()
    if len(angles_deg) == 0:
        return np.nan
    bins = np.linspace(0, 360, 17)
    binned = pd.cut(angles_deg, bins=bins, labels=COMPASS_DIRS, right=False, include_lowest=True)
    counts = binned.value_counts(normalize=True)
    return round(counts.max() * 100, 1)

# Simple 8-point compass (N, NE, E, SE, S, SW, W, NW) — easier to read than the full 16-point version
COMPASS_8 = ['N','NE','E','SE','S','SW','W','NW']

def deg_to_compass8(deg):
    """Convert a degree value (0-360) to the nearest of the 8 cardinal/intercardinal directions."""
    if pd.isna(deg):
        return None
    idx = int((deg / 45) + 0.5) % 8
    return COMPASS_8[idx]

def dominant_compass8(angles_deg):
    """Most frequent 8-point compass sector (simpler than the 16-point version)."""
    angles_deg = angles_deg.dropna()
    if len(angles_deg) == 0:
        return None
    bins = np.linspace(0, 360, 9)
    binned = pd.cut(angles_deg, bins=bins, labels=COMPASS_8, right=False, include_lowest=True)
    return binned.value_counts().idxmax()


# ─────────────────────────────────────────────
# 2. SUMMARY TABLES
# ─────────────────────────────────────────────
print("Computing summaries ...")

# Daily summary
daily = df.groupby("date").agg(
    temp_max   = ("temperature",           "max"),
    temp_min   = ("temperature",           "min"),
    temp_mean  = ("temperature",           "mean"),
    temp_range = ("temperature",           lambda x: x.max() - x.min()),
    hum_mean   = ("humidity",              "mean"),
    hum_min    = ("humidity",              "min"),
    hum_max    = ("humidity",              "max"),
    ghi_max    = ("ghi_solar_irradiance",  "max"),
    ghi_mean   = ("ghi_solar_irradiance",  "mean"),
    wind_mean  = ("wind_speed",            "mean"),
    wind_max   = ("wind_speed",            "max"),
    pressure   = ("atm_pressure",          "mean"),
).round(2)

daily.to_csv(os.path.join(OUTPUT_DIR, "daily_summary.csv"))

# ── Day vs Night summary (overall, across all days) ──
daynight_overall = df.groupby("period").agg(
    temp_mean = ("temperature",          "mean"),
    temp_max  = ("temperature",          "max"),
    temp_min  = ("temperature",          "min"),
    hum_mean  = ("humidity",             "mean"),
    hum_max   = ("humidity",             "max"),
    hum_min   = ("humidity",             "min"),
    ghi_mean  = ("ghi_solar_irradiance", "mean"),
    ghi_max   = ("ghi_solar_irradiance", "max"),
    wind_mean = ("wind_speed",           "mean"),
    wind_max  = ("wind_speed",           "max"),
    wind_min  = ("wind_speed",           "min"),
).round(2)

# GHI is meaningless at night (no sunlight) — drop it for that row
daynight_overall["ghi_mean"] = daynight_overall["ghi_mean"].astype(object)
daynight_overall["ghi_max"]  = daynight_overall["ghi_max"].astype(object)
daynight_overall.loc["Night", ["ghi_mean", "ghi_max"]] = "—"

# Circular mean wind direction + dominant compass sector, per period
wind_dir_summary = df.groupby("period")["wind_direction"].agg(
    wind_dir_mean_deg     = circular_mean_deg,
    wind_dir_dominant     = dominant_compass,       # 16-point, e.g. NNW, SSE
    wind_dir_dominant_8pt = dominant_compass8,      # simple 8-point, e.g. N, NE, S, SW
    wind_dir_dominant_pct = dominant_pct,           # % of readings in that dominant sector
)
wind_dir_summary["wind_dir_mean_compass"] = wind_dir_summary["wind_dir_mean_deg"].apply(deg_to_compass)
wind_dir_summary["wind_dir_mean_compass_8pt"] = wind_dir_summary["wind_dir_mean_deg"].apply(deg_to_compass8)
wind_dir_summary["wind_dir_mean_deg"] = wind_dir_summary["wind_dir_mean_deg"].round(1)

daynight_overall = daynight_overall.join(wind_dir_summary)
daynight_overall.to_csv(os.path.join(OUTPUT_DIR, "daynight_overall_summary.csv"))

print("\n── DAY vs NIGHT SUMMARY (overall) ─────────────────────")
print(daynight_overall.to_string())

# ── Day vs Night summary, broken down per day (date) ──
daynight_daily = df.groupby(["date", "period"]).agg(
    temp_mean = ("temperature",          "mean"),
    temp_max  = ("temperature",          "max"),
    temp_min  = ("temperature",          "min"),
    hum_mean  = ("humidity",             "mean"),
    hum_max   = ("humidity",             "max"),
    hum_min   = ("humidity",             "min"),
    ghi_mean  = ("ghi_solar_irradiance", "mean"),
    ghi_max   = ("ghi_solar_irradiance", "max"),
    wind_mean = ("wind_speed",           "mean"),
    wind_max  = ("wind_speed",           "max"),
).round(2).unstack("period")
daynight_daily.columns = [f"{stat}_{period.lower()}" for stat, period in daynight_daily.columns]

# Drop GHI columns for night (meaningless — no sunlight)
daynight_daily = daynight_daily.drop(columns=["ghi_mean_night", "ghi_max_night"])

# Per-day wind direction (circular mean + dominant compass), Day vs Night
wind_dir_daily = df.groupby(["date", "period"])["wind_direction"].agg(
    wind_dir_mean_deg     = circular_mean_deg,
    wind_dir_dominant     = dominant_compass,       # 16-point, e.g. NNW, SSE
    wind_dir_dominant_8pt = dominant_compass8,      # simple 8-point, e.g. N, NE, S, SW
    wind_dir_dominant_pct = dominant_pct,           # % of readings in that dominant sector
).unstack("period")
wind_dir_daily.columns = [f"{stat}_{period.lower()}" for stat, period in wind_dir_daily.columns]
wind_dir_daily["wind_dir_mean_deg_day"]   = wind_dir_daily["wind_dir_mean_deg_day"].round(1)
wind_dir_daily["wind_dir_mean_deg_night"] = wind_dir_daily["wind_dir_mean_deg_night"].round(1)

daynight_daily = daynight_daily.join(wind_dir_daily)
daynight_daily.to_csv(os.path.join(OUTPUT_DIR, "daynight_per_day_summary.csv"))

print("\n── DAY vs NIGHT SUMMARY (per day) ─────────────────────")
print(daynight_daily.to_string())

# Diurnal averages
diurnal = df.groupby("hour").agg(
    temp  = ("temperature",          "mean"),
    hum   = ("humidity",             "mean"),
    ghi   = ("ghi_solar_irradiance", "mean"),
    wind  = ("wind_speed",           "mean"),
).round(2)
diurnal.to_csv(os.path.join(OUTPUT_DIR, "hourly_diurnal.csv"))

# Overall stats
print("\n── OVERALL STATISTICS ────────────────────────────────")
stats_df = df[["temperature", "humidity", "ghi_solar_irradiance",
               "wind_speed", "atm_pressure"]].describe().round(2)
print(stats_df.to_string())

# Printable report table
with open(os.path.join(OUTPUT_DIR, "full_report_table.txt"), "w") as f:
    f.write(f"MICROCLIMATE SUMMARY REPORT\n{SITE_LABEL}\n")
    f.write("=" * 60 + "\n\n")
    f.write("DAILY SUMMARY TABLE\n")
    f.write(daily.to_string())
    f.write("\n\nOVERALL DESCRIPTIVE STATISTICS\n")
    f.write(stats_df.to_string())

# ─────────────────────────────────────────────
# 3. FIG 1 — DIURNAL PATTERN (Temp / Humidity / GHI)
# ─────────────────────────────────────────────
print("\nPlotting fig 1 – diurnal pattern ...")
fig, ax1 = plt.subplots(figsize=(11, 5))

ax1.fill_between(diurnal.index, diurnal["temp"], alpha=0.12, color=TEMP_COLOR)
ax1.plot(diurnal.index, diurnal["temp"], color=TEMP_COLOR, lw=2, label="Temperature (°C)")
ax1.set_xlabel("Hour of day", fontsize=11)
ax1.set_ylabel("Temperature (°C)", color=TEMP_COLOR, fontsize=11)
ax1.tick_params(axis="y", labelcolor=TEMP_COLOR)
ax1.set_xticks(range(0, 24, 2))
ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=30, ha="right")

ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)
ax2.plot(diurnal.index, diurnal["hum"], color=HUM_COLOR, lw=2,
         linestyle="--", label="Humidity (%)")
ax2.fill_between(diurnal.index, diurnal["hum"], alpha=0.07, color=HUM_COLOR)

# GHI on secondary axis (scaled)
ghi_vals = diurnal["ghi"].fillna(0)
ax2.plot(diurnal.index, ghi_vals / 10, color=GHI_COLOR, lw=1.8,
         linestyle=":", label="GHI ÷10 (W/m²)")
ax2.set_ylabel("Humidity (%) / GHI ÷10", fontsize=11)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

fig.suptitle(f"Average diurnal microclimate pattern\n{SITE_LABEL}", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig1_diurnal_pattern.png"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 4. FIG 2 — DAILY TEMPERATURE RANGE + HUMIDITY
# ─────────────────────────────────────────────
print("Plotting fig 2 – daily range ...")
dates     = list(daily.index)
date_nums = range(len(dates))

fig, ax1 = plt.subplots(figsize=(11, 5))

# Temperature range bars
ax1.bar(date_nums, daily["temp_max"], color=TEMP_COLOR, alpha=0.8,
        label="Max temp", zorder=3)
ax1.bar(date_nums, daily["temp_min"], color="#F0997B", alpha=0.8,
        label="Min temp", zorder=3)
ax1.axhline(0, color="#888", lw=0.8, linestyle="--")
ax1.set_ylabel("Temperature (°C)", fontsize=11, color=TEMP_COLOR)
ax1.tick_params(axis="y", labelcolor=TEMP_COLOR)
ax1.set_xticks(date_nums)
ax1.set_xticklabels([str(d) for d in dates], rotation=30, ha="right")

ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)
ax2.plot(date_nums, daily["hum_mean"], color=HUM_COLOR, lw=2,
         marker="o", ms=5, label="Mean humidity %")
ax2.set_ylabel("Mean humidity (%)", fontsize=11, color=HUM_COLOR)
ax2.tick_params(axis="y", labelcolor=HUM_COLOR)
ax2.set_ylim(0, 120)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

fig.suptitle(f"Daily temperature range and mean humidity\n{SITE_LABEL}", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig2_daily_range.png"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 4b. FIG 2b — DAY vs NIGHT COMPARISON (Temp & Humidity)
# ─────────────────────────────────────────────
print("Plotting fig 2b – day vs night comparison ...")
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

periods = ["Day", "Night"]
bar_colors = ["#F2A65A", "#2C3E66"]  # warm day color, cool night color

panel_specs = [
    (axes[0, 0], "temp_mean", "°C",  "Mean Temperature",  TEMP_COLOR),
    (axes[0, 1], "hum_mean",  "%",   "Mean Humidity",     HUM_COLOR),
    (axes[1, 1], "wind_mean", "m/s", "Mean Wind Speed",   "#5B7C99"),
]

for ax, col, unit, title, _ in panel_specs:
    vals = [daynight_overall.loc[p, col] for p in periods]
    max_v = max(vals) if max(vals) > 0 else 1
    ax.bar(periods, vals, color=bar_colors, width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + max_v * 0.03, f"{v:.1f}{unit}", ha="center",
                 fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(f"Mean ({unit})", fontsize=9)
    ax.set_ylim(0, max_v * 1.3)

# Add dominant wind direction (8-point, simpler) + % below the wind speed panel
for i, p in enumerate(periods):
    dom_dir = daynight_overall.loc[p, "wind_dir_dominant_8pt"]
    dom_pct = daynight_overall.loc[p, "wind_dir_dominant_pct"]
    x_frac = 0.27 if i == 0 else 0.73
    axes[1, 1].text(x_frac, -0.13, f"from {dom_dir} ({dom_pct:.0f}%)", transform=axes[1, 1].transAxes,
                     ha="center", fontsize=8.5, style="italic", color="#555")

# GHI panel — daytime only (night GHI is not meaningful, no sunlight)
day_ghi_mean = daynight_overall.loc["Day", "ghi_mean"]
day_ghi_max  = daily["ghi_max"].max() if "ghi_max" in daily else df["ghi_solar_irradiance"].max()
axes[1, 0].bar(["Day mean", "Day max"], [day_ghi_mean, day_ghi_max],
               color=[GHI_COLOR, "#FAC775"], width=0.5)
for i, v in enumerate([day_ghi_mean, day_ghi_max]):
    axes[1, 0].text(i, v + day_ghi_max * 0.03, f"{v:.0f} W/m²", ha="center",
                     fontsize=10, fontweight="bold")
axes[1, 0].set_title("Solar Irradiance (GHI) — daytime only", fontsize=11)
axes[1, 0].set_ylabel("W/m²", fontsize=9)
axes[1, 0].set_ylim(0, day_ghi_max * 1.3)
axes[1, 0].text(0.5, -0.18, "Night GHI omitted (no sunlight)", transform=axes[1, 0].transAxes,
                 ha="center", fontsize=8, style="italic", color="#888")

fig.suptitle(f"Day (06:00–18:00) vs Night (18:00–06:00)\n{SITE_LABEL}", fontsize=13)
fig.tight_layout()
fig.subplots_adjust(bottom=0.1)
fig.savefig(os.path.join(OUTPUT_DIR, "fig2b_day_vs_night.png"), bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────
# 5. FIG 3 — SOLAR RADIATION
# ─────────────────────────────────────────────
print("Plotting fig 3 – solar radiation ...")
fig, ax = plt.subplots(figsize=(11, 5))

x  = np.arange(len(dates))
w  = 0.4
ax.bar(x - w/2, daily["ghi_max"],  width=w, color=GHI_COLOR,   alpha=0.9, label="Daily max GHI")
ax.bar(x + w/2, daily["ghi_mean"], width=w, color="#FAC775",   alpha=0.9, label="Daily mean GHI")
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in dates], rotation=30, ha="right")
ax.set_ylabel("Solar irradiance (W/m²)", fontsize=11)
ax.legend(fontsize=9)
ax.set_title(f"Daily solar irradiance (GHI)\n{SITE_LABEL}", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig3_solar_radiation.png"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 6. FIG 4 — SOIL TEMPERATURE PROFILE
# ─────────────────────────────────────────────
print("Plotting fig 4 – soil temperature profile ...")
fig, ax = plt.subplots(figsize=(11, 5))

soil_diurnal = df.groupby("hour")[SOIL_COLS].mean()

for i, (col, lbl) in enumerate(zip(SOIL_COLS, SOIL_LABELS)):
    ax.plot(soil_diurnal.index, soil_diurnal[col],
            color=SOIL_COLORS[i % len(SOIL_COLORS)], lw=1.8,
            label=lbl, marker="o", ms=3)

ax.set_xlabel("Hour of day", fontsize=11)
ax.set_ylabel("Soil temperature (°C)", fontsize=11)
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=30, ha="right")
ax.legend(fontsize=8, ncol=2, loc="upper left")
ax.set_title(f"Diurnal soil temperature profile by depth\n{SITE_LABEL}", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig4_soil_temperature.png"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 7. FIG 5 — WIND ROSE
# ─────────────────────────────────────────────
print("Plotting fig 5 – wind rose ...")
wd    = df["wind_direction"].dropna()
ws    = df["wind_speed"].dropna()
valid = df[["wind_direction", "wind_speed"]].dropna()

n_bins   = 16
dir_bins = np.linspace(0, 360, n_bins + 1)
labels   = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]

fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111, projection="polar")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

dir_cut = pd.cut(valid["wind_direction"], bins=dir_bins, labels=False, right=False)
counts  = np.zeros(n_bins)
for i in range(n_bins):
    counts[i] = (dir_cut == i).sum()

theta  = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
width  = 2 * np.pi / n_bins
bars   = ax.bar(theta, counts, width=width * 0.85, bottom=0,
                color=TEMP_COLOR, alpha=0.75, edgecolor="white", lw=0.5)
ax.set_xticks(theta)
ax.set_xticklabels(labels, fontsize=8)
ax.set_title(f"Wind rose\n{SITE_LABEL}", fontsize=11, pad=20)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig5_wind_rose.png"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 8. FIG 6 — GHI vs TEMPERATURE CORRELATION
# ─────────────────────────────────────────────
print("Plotting fig 6 – GHI vs temperature correlation ...")
corr_df = df[["ghi_solar_irradiance", "temperature"]].dropna()
corr_df = corr_df[corr_df["ghi_solar_irradiance"] > 10]  # daytime only

slope, intercept, r_value, p_value, std_err = stats.linregress(
    corr_df["ghi_solar_irradiance"], corr_df["temperature"])

x_line = np.linspace(corr_df["ghi_solar_irradiance"].min(),
                      corr_df["ghi_solar_irradiance"].max(), 200)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(corr_df["ghi_solar_irradiance"], corr_df["temperature"],
           alpha=0.25, s=8, color=GHI_COLOR, label="Observations")
ax.plot(x_line, y_line, color=TEMP_COLOR, lw=2,
        label=f"Linear fit  r²={r_value**2:.3f}  p<0.001")
ax.set_xlabel("GHI solar irradiance (W/m²)", fontsize=11)
ax.set_ylabel("Temperature (°C)", fontsize=11)
ax.legend(fontsize=9)
ax.set_title(f"Solar irradiance vs. air temperature (daytime)\n{SITE_LABEL}", fontsize=12)

textstr = f"y = {slope:.4f}x + {intercept:.2f}\nr² = {r_value**2:.3f}"
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig6_correlation.png"), bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 9. DONE — PRINT SUMMARY
# ─────────────────────────────────────────────
print("\n── OUTPUT FILES ──────────────────────────────────────")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {OUTPUT_DIR}/{f}")

print(f"""
── KEY FINDINGS ──────────────────────────────────────
  Temperature  : {df['temperature'].min():.1f}°C (min) – {df['temperature'].max():.1f}°C (max)  | mean {df['temperature'].mean():.1f}°C
  Humidity     : {df['humidity'].min():.1f}% (min) – {df['humidity'].max():.1f}% (max)   | mean {df['humidity'].mean():.1f}%
  GHI peak     : {df['ghi_solar_irradiance'].max():.0f} W/m²  (mean daytime: {df[df['ghi_solar_irradiance']>0]['ghi_solar_irradiance'].mean():.0f} W/m²)
  Wind speed   : {df['wind_speed'].mean():.2f} m/s mean  |  {df['wind_speed'].max():.2f} m/s max
  GHI–Temp r²  : {r_value**2:.3f}  (slope: {slope:.4f} °C per W/m²)

  Day mean temp    : {daynight_overall.loc['Day','temp_mean']:.1f}°C  (max {daynight_overall.loc['Day','temp_max']:.1f}, min {daynight_overall.loc['Day','temp_min']:.1f})
  Night mean temp  : {daynight_overall.loc['Night','temp_mean']:.1f}°C  (max {daynight_overall.loc['Night','temp_max']:.1f}, min {daynight_overall.loc['Night','temp_min']:.1f})
  Day mean humidity   : {daynight_overall.loc['Day','hum_mean']:.1f}%  (max {daynight_overall.loc['Day','hum_max']:.1f}, min {daynight_overall.loc['Day','hum_min']:.1f})
  Night mean humidity : {daynight_overall.loc['Night','hum_mean']:.1f}%  (max {daynight_overall.loc['Night','hum_max']:.1f}, min {daynight_overall.loc['Night','hum_min']:.1f})
  Day mean GHI     : {daynight_overall.loc['Day','ghi_mean']:.1f} W/m²  (max {daynight_overall.loc['Day','ghi_max']:.1f})  [Night GHI omitted — no sunlight]
  Day mean wind    : {daynight_overall.loc['Day','wind_mean']:.2f} m/s  (max {daynight_overall.loc['Day','wind_max']:.2f})  |  mainly from {daynight_overall.loc['Day','wind_dir_dominant_8pt']} ({daynight_overall.loc['Day','wind_dir_dominant']}, {daynight_overall.loc['Day','wind_dir_dominant_pct']:.0f}% of readings)
  Night mean wind  : {daynight_overall.loc['Night','wind_mean']:.2f} m/s  (max {daynight_overall.loc['Night','wind_max']:.2f})  |  mainly from {daynight_overall.loc['Night','wind_dir_dominant_8pt']} ({daynight_overall.loc['Night','wind_dir_dominant']}, {daynight_overall.loc['Night','wind_dir_dominant_pct']:.0f}% of readings)
──────────────────────────────────────────────────────
All figures saved to ./{OUTPUT_DIR}/
""")

# ─────────────────────────────────────────────
# EXTENSION: THREE-TREATMENT COMPARISON
# ─────────────────────────────────────────────
# When you have T1, T2, T3 data, load them all and use this block:
#
# from scipy.stats import f_oneway
#
# t1 = pd.read_excel("T1_data.xlsx"); t1['treatment'] = 'T1'
# t2 = pd.read_excel("T2_data.xlsx"); t2['treatment'] = 'T2'
# t3 = pd.read_excel("T3_data.xlsx"); t3['treatment'] = 'T3'
# combined = pd.concat([t1, t2, t3])
#
# # One-way ANOVA on temperature
# f_stat, p_val = f_oneway(t1['temperature'].dropna(),
#                           t2['temperature'].dropna(),
#                           t3['temperature'].dropna())
# print(f"ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
#
# # Delta-T (indoor minus outdoor)
# outdoor = pd.read_excel("outdoor.xlsx")
# t1['delta_T'] = t1['temperature'].values - outdoor['temperature'].values
#
# # Overlay diurnal plot for all treatments
# fig, ax = plt.subplots(figsize=(11,5))
# for label, data, color in [('T1',t1,'#D85A30'),('T2',t2,'#185FA5'),('T3',t3,'#3B6D11')]:
#     d = data.groupby(data['recorded_at'].dt.hour)['temperature'].mean()
#     ax.plot(d.index, d.values, color=color, lw=2, label=label)
# ax.legend(); ax.set_xlabel("Hour"); ax.set_ylabel("Temperature (°C)")
# plt.savefig("output/fig_treatment_comparison.png", bbox_inches='tight')
