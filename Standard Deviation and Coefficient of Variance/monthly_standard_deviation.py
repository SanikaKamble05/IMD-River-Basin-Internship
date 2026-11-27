import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import os

# FILE PATHS
csv_path = (
    r"E:\IMD Project\Final Maps\basin_series_nonacc.csv"
)

shapefile_path = (
    r"E:\IMD Project\Final Maps\SUB_BASIN_220"
    r"\SUB_BASIN_220.shp"
)

india_path = (
    r"E:\IMD Project\Final Maps\India outline boundary"
    r"\India outline boundary\India outline boundary.shp"
)

output_csv = (
    r"E:\IMD Project\Final Maps"
    r"\river_basin_annual_standard_deviation_1971_2020.csv"
)

output_shapefile = (
    r"E:\IMD Project\Final Maps\SUB_BASIN_220"
    r"\SUB_BASIN_220_annual_standard_deviation.shp"
)

output_map = (
    r"E:\IMD Project\Final Maps"
    r"\river_basin_annual_standard_deviation_1971_2020.png"
)

# READ CSV
df = pd.read_csv(csv_path)

print("CSV loaded successfully.")
print("Total rows:", len(df))

# SELECT 1971-2020
df["YEAR"] = pd.to_numeric(
    df["YEAR"],
    errors="coerce"
)

df = df[
    (df["YEAR"] >= 1971) &
    (df["YEAR"] <= 2020)
].copy()

# CLEAN NAMES
df["basin_name"] = (
    df["basin_name"]
    .astype(str)
    .str.strip()
)

df["subbasin_name"] = (
    df["subbasin_name"]
    .astype(str)
    .str.strip()
)

# IDENTIFY DAILY RAINFALL COLUMNS
metadata_columns = [
    "basin_name",
    "subbasin_id",
    "subbasin_name",
    "YEAR",
    "STATION"
]

daily_columns = [
    col for col in df.columns
    if col not in metadata_columns
    and not col.startswith("Unnamed")
]

print(
    "\nNumber of daily rainfall columns:",
    len(daily_columns)
)

# CONVERT RAINFALL TO NUMERIC
df[daily_columns] = df[daily_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# CALCULATE ANNUAL RAINFALL
df["Annual_Rainfall"] = (
    df[daily_columns]
    .sum(
        axis=1,
        skipna=True
    )
)

# CREATE ANNUAL DATA
annual_data = df[
    [
        "basin_name",
        "subbasin_id",
        "subbasin_name",
        "YEAR",
        "Annual_Rainfall"
    ]
].copy()

# CALCULATE STANDARD DEVIATION
sd_data = (
    annual_data
    .groupby(
        [
            "basin_name",
            "subbasin_id",
            "subbasin_name"
        ],
        dropna=False
    )
    .agg(
        Standard_Deviation=(
            "Annual_Rainfall",
            lambda x: np.std(x, ddof=0)
        ),
        Number_of_Years=(
            "Annual_Rainfall",
            "count"
        )
    )
    .reset_index()
)

sd_data["Standard_Deviation"] = (
    sd_data["Standard_Deviation"]
    .round(2)
)

sd_data["Map_ID"] = (
    sd_data["subbasin_id"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .str.zfill(9)
)


# ============================================================
# SAVE CSV
# ============================================================

os.makedirs(
    os.path.dirname(output_csv),
    exist_ok=True
)

sd_data.to_csv(
    output_csv,
    index=False
)


# ============================================================
# STANDARD DEVIATION INFORMATION
# ============================================================

print("\n======================================")
print("STANDARD DEVIATION INFORMATION")
print("======================================")

print(
    "Number of sub-basins:",
    len(sd_data)
)

print(
    "Number of years:",
    df["YEAR"].nunique()
)

print(
    "Minimum year:",
    int(df["YEAR"].min())
)

print(
    "Maximum year:",
    int(df["YEAR"].max())
)


min_sd = (
    sd_data["Standard_Deviation"]
    .min()
)

max_sd = (
    sd_data["Standard_Deviation"]
    .max()
)


min_row = sd_data.loc[
    sd_data["Standard_Deviation"].idxmin()
]

max_row = sd_data.loc[
    sd_data["Standard_Deviation"].idxmax()
]


# ============================================================
# LOWEST STANDARD DEVIATION
# ============================================================

print("\n======================================")
print("LOWEST STANDARD DEVIATION")
print("======================================")

print(
    "Basin Name:",
    min_row["basin_name"]
)

print(
    "Sub-basin ID:",
    min_row["subbasin_id"]
)

print(
    "Sub-basin Name:",
    min_row["subbasin_name"]
)

print(
    "Minimum SD:",
    min_sd,
    "mm"
)


# ============================================================
# HIGHEST STANDARD DEVIATION
# ============================================================

print("\n======================================")
print("HIGHEST STANDARD DEVIATION")
print("======================================")

print(
    "Basin Name:",
    max_row["basin_name"]
)

print(
    "Sub-basin ID:",
    max_row["subbasin_id"]
)

print(
    "Sub-basin Name:",
    max_row["subbasin_name"]
)

print(
    "Maximum SD:",
    max_sd,
    "mm"
)


# ============================================================
# FIRST 10 SUB-BASINS
# ============================================================

print("\n======================================")
print("FIRST 10 SUB-BASINS")
print("======================================")

print(
    sd_data[
        [
            "basin_name",
            "subbasin_id",
            "subbasin_name",
            "Standard_Deviation",
            "Number_of_Years"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# READ SHAPEFILES
# ============================================================

basins = gpd.read_file(
    shapefile_path
)

india = gpd.read_file(
    india_path
)


# ============================================================
# CREATE MAP ID FROM SHAPEFILE
# ============================================================

basins["Map_ID"] = (
    basins["UIN"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .str.zfill(9)
)


# ============================================================
# MATCH CSV AND SHAPEFILE
# ============================================================

csv_ids = set(
    sd_data["Map_ID"]
)

basins = basins[
    basins["Map_ID"].isin(csv_ids)
].copy()


print("\n======================================")
print("MAP MATCHING")
print("======================================")

print(
    "Sub-basins in CSV:",
    len(csv_ids)
)

print(
    "Sub-basins matched with shapefile:",
    len(basins)
)


# ============================================================
# MATCH CRS
# ============================================================

india = india.to_crs(
    basins.crs
)


# ============================================================
# CLIP TO INDIA
# ============================================================

basins = gpd.clip(
    basins,
    india
)


# ============================================================
# KEEP POLYGON GEOMETRIES
# ============================================================

basins = basins[
    basins.geometry.geom_type.isin(
        [
            "Polygon",
            "MultiPolygon"
        ]
    )
].copy()


# ============================================================
# CONVERT TO WGS84
# ============================================================

basins = basins.to_crs(
    epsg=4326
)

india = india.to_crs(
    epsg=4326
)


# ============================================================
# MERGE STANDARD DEVIATION
# ============================================================

basins = basins.merge(
    sd_data[
        [
            "Map_ID",
            "Standard_Deviation",
            "Number_of_Years"
        ]
    ],
    on="Map_ID",
    how="inner"
)


# ============================================================
# SAVE SHAPEFILE
# ============================================================

basins.to_file(
    output_shapefile
)


# ============================================================
# COLORBAR CLASS BOUNDARIES
#
# 0-1400 mm
# Difference between labels = 100 mm
#
# Values greater than 1400 mm
# are represented by the triangle
# at the top of the colorbar.
# ============================================================

bounds = np.arange(
    0,
    1401,
    50
)


# Number of color intervals
number_of_classes = (
    len(bounds) - 1
)


# ============================================================
# VIRIDIS_R WITH DISCRETE COLORS
# ============================================================

base_cmap = plt.get_cmap(
    "viridis_r"
)

colors = base_cmap(
    np.linspace(
        0,
        1,
        number_of_classes
    )
)

cmap = ListedColormap(
    colors
)


# ============================================================
# DISCRETE COLOR NORMALIZATION
# ============================================================

norm = BoundaryNorm(
    bounds,
    ncolors=number_of_classes,
    clip=True
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 14)
)


# ============================================================
# DATA FOR MAP
# ============================================================

data = basins[
    basins["Standard_Deviation"].notna()
].copy()


# ============================================================
# PLOT MAP
# ============================================================

data.plot(
    column="Standard_Deviation",
    cmap=cmap,
    norm=norm,
    linewidth=0.35,
    edgecolor="black",
    ax=ax,
    legend=False
)


# ============================================================
# INDIA OUTLINE
# ============================================================

india.boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.2
)


# ============================================================
# COLORBAR
# ============================================================

sm = ScalarMappable(
    norm=norm,
    cmap=cmap
)

sm.set_array([])


cbar = fig.colorbar(
    sm,
    ax=ax,
    fraction=0.035,
    pad=0.02,
    boundaries=bounds,
    spacing="uniform",
    extend="max",
    extendfrac="auto"
)


# ============================================================
# COLORBAR TICKS
#
# 0, 100, 200, ..., 1400
# ============================================================

colorbar_ticks = np.arange(
    0,
    1401,
    100
)


cbar.set_ticks(
    colorbar_ticks
)


colorbar_labels = [
    str(int(value))
    for value in colorbar_ticks
]


cbar.set_ticklabels(
    colorbar_labels
)

# COLORBAR TITLE
cbar.set_label(
    "Annual Rainfall Standard Deviation (mm)",
    fontsize=18,
    labelpad=30,
    fontweight="bold"
)

# COLORBAR TICK STYLE
cbar.ax.tick_params(
    labelsize=20,
    length=6,
    width=1.5
)

ax.set_title(
    "INDIA METEOROLOGICAL DEPARTMENT\n"
    "ANNUAL RAINFALL STANDARD DEVIATION\n"
    "FOR THE PERIOD 1971 - 2020",
    fontsize=18,
    fontweight="bold",
    pad=15
)

# GRID
ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)

# REMOVE AXIS LABELS
ax.set_xlabel("")
ax.set_ylabel("")

# INCREASE LATITUDE AND LONGITUDE FONT SIZE
ax.tick_params(
    axis="both",
    which="major",
    labelsize=22,
    length=10,
    width=2,
    direction="out",
    colors="black"
)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(22)
    label.set_fontweight("normal")
    label.set_color("black")

# LAYOUT
plt.tight_layout()

# SAVE MAP
plt.savefig(
    output_map,
    dpi=300,
    bbox_inches="tight"
)


plt.show()

# FINAL OUTPUT
print("\n======================================")
print("ANNUAL STANDARD DEVIATION COMPLETED")
print("======================================")

print("\nCSV:")
print(output_csv)

print("\nShapefile:")
print(output_shapefile)

print("\nMap:")
print(output_map)

# ACTUAL SD RANGE
print("\n======================================")
print("ACTUAL SD RANGE")
print("======================================")

print(
    "Minimum:",
    min_sd,
    "mm"
)

print(
    "Maximum:",
    max_sd,
    "mm"
)

print(
    "\nNumber of color classes:",
    number_of_classes
)