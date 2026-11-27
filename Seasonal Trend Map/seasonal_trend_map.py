import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.lines import Line2D
from pathlib import Path

# ==========================================================
# 1. FILE PATHS
# ==========================================================

base_path = Path(r"E:\IMD Project\Final Maps")

monthly_csv = (
    base_path / "river_basin_monthly_rainfall_sum.csv"
)

shapefile = (
    base_path / "SUB_BASIN_220" / "SUB_BASIN_220.shp"
)

india_path = (
    base_path
    / "India outline boundary"
    / "India outline boundary"
    / "India outline boundary.shp"
)

trend_csv = (
    base_path / "river_basin_seasonal_rainfall_trend_1971_2020.csv"
)

output_map = (
    base_path / "Seasonal_Rainfall_Trend_Maps_1971_2020.png"
)

# ==========================================================
# 2. DEFINE MONTHS AND SEASONS
# ==========================================================

months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

seasons = {
    "JF": ["January", "February"],
    "MAM": ["March", "April", "May"],
    "JJAS": ["June", "July", "August", "September"],
    "OND": ["October", "November", "December"]
}

# ==========================================================
# 3. READ MONTHLY RAINFALL DATA
# ==========================================================

df = pd.read_csv(monthly_csv)

required_columns = [
    "River Basin",
    "Sub Basin ID",
    "Sub Basin Name",
    "Year"
] + months

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in monthly CSV: {missing_columns}"
    )

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

for month in months:
    df[month] = pd.to_numeric(
        df[month],
        errors="coerce"
    )

df = df[
    df["Year"].between(1971, 2020)
].copy()

df = df.dropna(
    subset=["Year", "Sub Basin ID"]
)

df["Year"] = df["Year"].astype(int)

# Standardize sub-basin IDs
df["Sub Basin ID"] = (
    df["Sub Basin ID"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
)

df = df.dropna(
    subset=["Sub Basin ID"]
)

# ==========================================================
# 4. CREATE SUB-BASIN INFORMATION TABLE
# ==========================================================

subbasins = (
    df[
        [
            "River Basin",
            "Sub Basin ID",
            "Sub Basin Name"
        ]
    ]
    .drop_duplicates(
        subset=["Sub Basin ID"],
        keep="first"
    )
    .reset_index(drop=True)
)

print("\n==========================================")
print("SEASONAL RAINFALL TREND ANALYSIS")
print("==========================================")
print("Years:", df["Year"].min(), "to", df["Year"].max())
print("Unique sub-basins:", len(subbasins))

# ==========================================================
# 5. CALCULATE ANNUAL SEASONAL RAINFALL TOTALS
# ==========================================================

# If a sub-basin has multiple records for the same year,
# combine its monthly values before calculating seasonal totals.
annual_monthly = (
    df.groupby(
        [
            "River Basin",
            "Sub Basin ID",
            "Sub Basin Name",
            "Year"
        ],
        as_index=False
    )[months]
    .sum(min_count=1)
)

seasonal_data = annual_monthly[
    [
        "River Basin",
        "Sub Basin ID",
        "Sub Basin Name",
        "Year"
    ]
].copy()

for season, season_months in seasons.items():
    seasonal_data[season] = (
        annual_monthly[season_months]
        .sum(axis=1, min_count=len(season_months))
    )

# ==========================================================
# 6. CALCULATE TREND FOR EACH SUB-BASIN AND SEASON
# ==========================================================

seasonal_results = {}

all_results = []

for season, season_months in seasons.items():

    results = []

    for _, sub in subbasins.iterrows():

        sub_id = sub["Sub Basin ID"]

        sub_data = seasonal_data[
            seasonal_data["Sub Basin ID"] == sub_id
        ].copy()

        sub_data = sub_data.sort_values("Year")

        sub_data = sub_data.dropna(
            subset=[season]
        )

        x = sub_data["Year"].to_numpy(dtype=float)
        y = sub_data[season].to_numpy(dtype=float)

        if len(sub_data) < 2 or np.unique(x).size < 2:
            slope = np.nan
            intercept = np.nan
            p_value = np.nan
            category = "No Data"

        else:
            regression = linregress(x, y)

            slope = regression.slope
            intercept = regression.intercept
            p_value = regression.pvalue

            if p_value < 0.05 and slope > 0:
                category = "Significantly Increasing"

            elif p_value < 0.05 and slope < 0:
                category = "Significantly Decreasing"

            elif slope > 0:
                category = "Increasing"

            elif slope < 0:
                category = "Decreasing"

            else:
                category = "No Data"

        result = {
            "Season": season,
            "River Basin": sub["River Basin"],
            "Sub Basin ID": sub_id,
            "Sub Basin Name": sub["Sub Basin Name"],
            "Start Year": (
                int(sub_data["Year"].min())
                if not sub_data.empty else np.nan
            ),
            "End Year": (
                int(sub_data["Year"].max())
                if not sub_data.empty else np.nan
            ),
            "Number of Years": len(sub_data),
            "Trend Slope": slope,
            "Intercept": intercept,
            "P-Value": p_value,
            "Trend Category": category
        }

        results.append(result)
        all_results.append(result)

    seasonal_results[season] = pd.DataFrame(results)

# ==========================================================
# 7. SAVE SEASONAL TREND RESULTS
# ==========================================================

trend_df = pd.DataFrame(all_results)

trend_df.to_csv(
    trend_csv,
    index=False
)

print("\nSeasonal trend categories:")

for season in seasons:
    print(f"\n{season}:")
    print(
        seasonal_results[season]["Trend Category"]
        .value_counts()
    )

print("\nTrend CSV saved to:")
print(trend_csv)

# ==========================================================
# 8. READ SUB-BASIN SHAPEFILE AND INDIA OUTLINE
# ==========================================================

basins = gpd.read_file(shapefile)
india = gpd.read_file(india_path)

if "UIN" not in basins.columns:
    raise ValueError(
        "The shapefile does not contain the required 'UIN' column."
    )

# Standardize shapefile IDs
basins["Map_ID"] = (
    basins["UIN"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .str.zfill(9)
)

# ==========================================================
# 9. CLIP BASINS TO INDIA
# ==========================================================

india = india.to_crs(basins.crs)

basins = gpd.clip(
    basins,
    india
)

# Keep polygon geometries only
basins = basins[
    basins.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )
].copy()

# Use geographic coordinates for consistent map plotting
basins = basins.to_crs(epsg=4326)
india = india.to_crs(epsg=4326)

# ==========================================================
# 10. DEFINE TREND COLORS
# ==========================================================

category_colors = {
    "No Data": "white",
    "Significantly Increasing": "blue",
    "Increasing": "lightblue",
    "Decreasing": "lightcoral",
    "Significantly Decreasing": "red"
}

# ==========================================================
# 11. CREATE LARGE 2 × 2 SEASONAL MAP FIGURE
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(28, 30)
)

fig.suptitle(
    "SEASONAL RAINFALL TREND MAP\n"
    "FOR THE PERIOD 1971–2020",
    fontsize=34,
    fontweight="bold",
    y=0.985,
    linespacing=1.15
)

# Map order
season_order = ["JF", "MAM", "JJAS", "OND"]

for ax, season in zip(axes.flat, season_order):

    season_trend_df = seasonal_results[season].copy()

    # Create standardized map IDs
    season_trend_df["Map_ID"] = (
        season_trend_df["Sub Basin ID"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .str.zfill(9)
    )

    # Merge seasonal trend categories with the geometries
    season_map = basins.drop(
        columns=["Trend Category"],
        errors="ignore"
    ).copy()

    season_map = season_map.merge(
        season_trend_df[
            ["Map_ID", "Trend Category"]
        ],
        on="Map_ID",
        how="left"
    )

    season_map["Trend Category"] = (
        season_map["Trend Category"]
        .fillna("No Data")
    )

    # Plot each category
    for category, color in category_colors.items():

        selected = season_map[
            season_map["Trend Category"] == category
        ]

        if not selected.empty:
            selected.plot(
                ax=ax,
                color=color,
                edgecolor="black",
                linewidth=0.5
            )

    # India outline
    india.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.3
    )

    # Seasonal title
    ax.set_title(
        f"{season}",
        fontsize=25,
        fontweight="bold",
        pad=16
    )

    # Remove axis labels only
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # Increase latitude and longitude font size
    ax.tick_params(
        axis="both",
        which="major",
        labelbottom=True,
        labelleft=True,
        bottom=True,
        left=True,
        labelsize=30,
        length=8,
        width=1.5,
        direction="out",
        colors="black"
    )
    
    # Keep tick labels normal (not bold)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(30)
        label.set_fontweight("normal")
        label.set_color("black")
        
        # Keep a subtle grid
        ax.grid(
            True,
            linestyle="--",
            linewidth=0.4,
            alpha=0.25
        )

    # Maintain equal geographic scale
    ax.set_aspect("equal", adjustable="box")

# ==========================================================
# 12. SHARED RAINFALL TREND LEGEND
# ==========================================================

legend_elements = [
    Line2D(
        [0], [0],
        marker="s",
        linestyle="None",
        color="none",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=18,
        label="No Data"
    ),
    Line2D(
        [0], [0],
        marker="s",
        linestyle="None",
        color="none",
        markerfacecolor="blue",
        markeredgecolor="black",
        markersize=18,
        label="Significantly Increasing"
    ),
    Line2D(
        [0], [0],
        marker="s",
        linestyle="None",
        color="none",
        markerfacecolor="lightblue",
        markeredgecolor="black",
        markersize=18,
        label="Increasing"
    ),
    Line2D(
        [0], [0],
        marker="s",
        linestyle="None",
        color="none",
        markerfacecolor="lightcoral",
        markeredgecolor="black",
        markersize=18,
        label="Decreasing"
    ),
    Line2D(
        [0], [0],
        marker="s",
        linestyle="None",
        color="none",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=18,
        label="Significantly Decreasing"
    )
]

fig.legend(
    handles=legend_elements,
    title="RAINFALL TREND",
    title_fontsize=23,
    fontsize=20,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    frameon=True,
    fancybox=True,
    framealpha=1,
    edgecolor="lightgray",
    facecolor="white",
    borderpad=1.5,
    labelspacing=1.1,
    columnspacing=2.5,
    handletextpad=0.8
)

# Leave space below the maps for the legend
fig.subplots_adjust(bottom=0.18)

plt.savefig(output_map, dpi=300, bbox_inches="tight")

# ==========================================================
# 13. ADJUST LAYOUT
# ==========================================================

plt.subplots_adjust(
    left=0.035,
    right=0.965,
    top=0.925,
    bottom=0.095,
    wspace=0.10,
    hspace=0.15
)

# ==========================================================
# 14. SAVE FIGURE
# ==========================================================

plt.savefig(
    output_map,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# 15. FINAL OUTPUT
# ==========================================================

print("\n==========================================")
print("SEASONAL RAINFALL TREND MAP COMPLETED")
print("==========================================")
print("Seasons plotted:", ", ".join(season_order))
print("Period: 1971–2020")
print("Sub-basins:", len(subbasins))
print("Trend records:", len(trend_df))
print("Trend CSV:", trend_csv)
print("Output map:", output_map)
print("==========================================")
