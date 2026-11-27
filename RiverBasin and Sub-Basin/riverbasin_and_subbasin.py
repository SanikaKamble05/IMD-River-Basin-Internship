import geopandas as gpd
import matplotlib.pyplot as plt

# 1. Read river sub-basin shapefile
basins = gpd.read_file(
    r"E:\IMD Project\Final Maps\SUB_BASIN_220\SUB_BASIN_220.shp"
)

# 2. Read India outline shapefile
india = gpd.read_file(
    r"E:\IMD Project\Final Maps\India outline boundary"
    r"\India outline boundary"
    r"\India outline boundary.shp"
)

# 3. Convert India outline to same CRS as basins
india = india.to_crs(basins.crs)

# 4. Clip sub-basins to India boundary
basins = gpd.clip(basins, india)

# 5. Merge sub-basins into river basins
river_basins = basins.dissolve(
    by="BASIN"
).reset_index()

# 6. Convert both datasets to Latitude/Longitude
basins = basins.to_crs(epsg=4326)
river_basins = river_basins.to_crs(epsg=4326)
india = india.to_crs(epsg=4326)

# 7. Create side-by-side plots
fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 10)
)

# Left: River Basins
river_basins.plot(
    ax=axes[0],
    column="BASIN",
    cmap="tab20",
    edgecolor="black",
    linewidth=0.8
)

# India outer boundary
india.boundary.plot(
    ax=axes[0],
    edgecolor="black",
    linewidth=2.5
)

axes[0].set_title(
    "River Basins of India",
    fontsize=16,
    fontweight="bold"
)

# Latitude and longitude tick formatting
axes[0].tick_params(
    axis="both",
    which="major",
    labelsize=18,
    length=12,
    width=2.5,
    direction="out",
    colors="black"
)

for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
    label.set_fontsize(18)
    label.set_fontweight("bold")
    label.set_color("black")


# Right: River Sub-Basins
basins.plot(
    ax=axes[1],
    column="BASIN",
    cmap="tab20",
    edgecolor="black",
    linewidth=0.4
)

# India outer boundary
india.boundary.plot(
    ax=axes[1],
    edgecolor="black",
    linewidth=2.5
)

axes[1].set_title(
    "River Sub-Basins of India",
    fontsize=16,
    fontweight="bold"
)

# Latitude and longitude tick formatting
axes[1].tick_params(
    axis="both",
    which="major",
    labelsize=18,
    length=12,
    width=2.5,
    direction="out",
    colors="black"
)

for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
    label.set_fontsize(18)
    label.set_fontweight("bold")
    label.set_color("black")


# 8. Adjust layout
plt.tight_layout()

plt.savefig(
    r"E:\IMD Project\Final Maps\River_Basins_and_Sub_Basins.png",
    dpi=300,
    bbox_inches="tight",
    format="png"
)

# 9. Display
plt.show()
