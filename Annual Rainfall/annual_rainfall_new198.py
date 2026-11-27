import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable


# ============================================================
# FILE PATHS
# ============================================================

shapefile_path = (
    r"E:\IMD Project\Final Maps\SUB_BASIN_220"
    r"\SUB_BASIN_220.shp"
)

india_path = (
    r"E:\IMD Project\Final Maps\India outline boundary"
    r"\India outline boundary"
    r"\India outline boundary.shp"
)

# NEW DATASET
csv_path = (
    r"E:\IMD Project\Final Maps"
    r"\Subbasin_Normals_2026_198.csv"
)

output_shapefile = (
    r"E:\IMD Project\Final Maps\SUB_BASIN_220"
    r"\SUB_BASIN_220_annual_rainfall.shp"
)

output_csv = (
    r"E:\IMD Project\Final Maps"
    r"\river_basin_annual_rainfall.csv"
)

output_map = (
    r"E:\IMD Project\Final Maps"
    r"\river_basin_annual_rainfall.png"
)


# ============================================================
# READ SHAPEFILE
# ============================================================

basins = gpd.read_file(
    shapefile_path
)


# ============================================================
# CREATE ID FROM SHAPEFILE UIN
# ============================================================

basins["ID"] = (
    basins["UIN"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .str.zfill(9)
)


# ============================================================
# READ NEW CSV
# ============================================================

df = pd.read_csv(
    csv_path
)


# ============================================================
# CREATE ID FROM NEW CSV UIN
# ============================================================

df["ID"] = (
    df["UIN"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .str.zfill(9)
)


# ============================================================
# CONVERT ANNUAL COLUMN TO NUMERIC
# ============================================================

df["annual"] = pd.to_numeric(
    df["annual"],
    errors="coerce"
)


# ============================================================
# SELECT ID AND ANNUAL RAINFALL
# ============================================================

rainfall_data = df[
    [
        "ID",
        "annual"
    ]
].copy()


# ============================================================
# MERGE ANNUAL RAINFALL WITH SHAPEFILE
# ============================================================

basins = basins.merge(
    rainfall_data,
    on="ID",
    how="left"
)


# ============================================================
# MATCHING INFORMATION
# ============================================================

matched = basins["annual"].notna().sum()

unmatched = basins["annual"].isna().sum()


print("==========================================")
print("ANNUAL RAINFALL MATCHING")
print("==========================================")

print(
    f"Total shapefile basins      : {len(basins)}"
)

print(
    f"IDs matching CSV            : {matched}"
)

print(
    f"IDs not matching            : {unmatched}"
)

print(
    f"Basins with annual rainfall : {matched}"
)

print(
    f"Basins without rainfall     : {unmatched}"
)

print("==========================================")


# ============================================================
# CREATE OUTPUT CSV
# ============================================================

annual_output = basins[
    [
        "ID",
        "BASIN",
        "SUBBASIN",
        "annual"
    ]
].copy()


annual_output.to_csv(
    output_csv,
    index=False
)


# ============================================================
# KEEP ONLY POLYGON AND MULTIPOLYGON
# ============================================================

basins = basins[
    basins.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )
].copy()


# ============================================================
# READ INDIA OUTLINE
# ============================================================

india = gpd.read_file(
    india_path
)


# ============================================================
# MATCH CRS
# ============================================================

india = india.to_crs(
    basins.crs
)


# ============================================================
# CLIP BASINS TO INDIA
# ============================================================

basins = gpd.clip(
    basins,
    india
)


# ============================================================
# REMOVE INVALID GEOMETRY TYPES
# ============================================================

basins = basins[
    basins.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )
].copy()


# ============================================================
# SAVE UPDATED SHAPEFILE
# ============================================================

basins.to_file(
    output_shapefile,
    driver="ESRI Shapefile"
)


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
# ANNUAL RAINFALL STATISTICS
# ============================================================

rain_min = basins["annual"].min()

rain_max = basins["annual"].max()


print()
print("==========================================")
print("ANNUAL RAINFALL STATISTICS")
print("==========================================")

print(
    f"Minimum annual rainfall : {rain_min:.2f} mm"
)

print(
    f"Maximum annual rainfall : {rain_max:.2f} mm"
)

print("==========================================")


# ============================================================
# RAINFALL COLOR BOUNDARIES
# ============================================================

rainfall_boundaries = list(
    range(
        0,
        4001,
        200
    )
)


# ============================================================
# NUMBER OF COLOR CLASSES
# ============================================================

num_classes = (
    len(rainfall_boundaries) - 1
)


# ============================================================
# COLORMAP
# ============================================================

cmap = plt.get_cmap(
    "viridis_r",
    num_classes
)


# ============================================================
# BOUNDARY NORMALIZATION
# ============================================================

norm = BoundaryNorm(
    rainfall_boundaries,
    cmap.N,
    clip=True
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 16)
)


# ============================================================
# PLOT BASINS WITH ANNUAL DATA
# ============================================================

basins_with_data = basins[
    basins["annual"].notna()
].copy()


if not basins_with_data.empty:

    basins_with_data.plot(
        ax=ax,
        column="annual",
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.45
    )


# ============================================================
# INDIA OUTLINE
# ============================================================

india.boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.5
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "Annual Normal Rainfall of River Sub-Basins of India",
    fontsize=22,
    fontweight="bold",
    pad=18
)


# ============================================================
# REMOVE AXIS LABELS
# ============================================================

ax.set_xlabel("")
ax.set_ylabel("")


# ============================================================
# AXIS TICK SIZE
# ============================================================

ax.tick_params(
    axis="both",
    labelsize=22,
    length=8,
    width=1.5
)


# ============================================================
# CREATE COLORBAR
# ============================================================

sm = ScalarMappable(
    cmap=cmap,
    norm=norm
)

sm.set_array([])


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(
    sm,
    ax=ax,
    boundaries=rainfall_boundaries,
    spacing="uniform",
    fraction=0.055,
    pad=0.04
)


# ============================================================
# COLORBAR TICKS
# ============================================================

cbar.set_ticks(
    rainfall_boundaries
)


cbar.set_ticklabels(
    [
        str(value)
        for value in rainfall_boundaries
    ]
)


# ============================================================
# COLORBAR TICK APPEARANCE
# ============================================================

cbar.ax.tick_params(
    labelsize=16,
    colors="black",
    length=9,
    width=1.8
)


for label in cbar.ax.get_yticklabels():

    label.set_fontsize(16)

    label.set_fontweight("bold")

    label.set_color("black")


# ============================================================
# COLORBAR LABEL
# ============================================================

cbar.set_label(
    "Rainfall (mm)",
    fontsize=19,
    fontweight="bold",
    color="black",
    labelpad=22
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# SAVE MAP
# ============================================================

plt.savefig(
    output_map,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# SHOW MAP
# ============================================================

plt.show()


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print()
print("==========================================")
print("PROCESS COMPLETED")
print("==========================================")

print(
    f"Annual CSV saved        : {output_csv}"
)

print(
    f"Updated shapefile saved : {output_shapefile}"
)

print(
    f"Map saved               : {output_map}"
)

print("==========================================")