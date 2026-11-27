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
basins = gpd.clip(
    basins,
    india
)

# 5. Create river basin boundaries
river_basins = basins.dissolve(
    by="BASIN"
).reset_index()

# 6. Convert to Latitude/Longitude
basins = basins.to_crs(
    epsg=4326
)

river_basins = river_basins.to_crs(
    epsg=4326
)

india = india.to_crs(
    epsg=4326
)

# 7. Create one map
fig, ax = plt.subplots(
    figsize=(10, 10)
)

# 8. Plot sub-basins with different colors
basins.plot(
    ax=ax,
    column="BASIN",
    cmap="tab20",
    edgecolor="black",
    linewidth=0.4
)

# 9. Plot river basin boundaries with thick dark outline
river_basins.boundary.plot(
    ax=ax,
    color="black",
    linewidth=2.0
)

# 10. Plot India outer boundary
india.boundary.plot(
    ax=ax,
    color="black",
    linewidth=2.5
)

# 11. Title
ax.set_title(
    "River Basins and Sub-Basins of India",
    fontsize=16,
    fontweight="bold",
    pad=12
)

# 12. Keep latitude and longitude numbers
ax.tick_params(
    axis="both",
    labelsize=10
)

# 13. Remove axis labels
ax.set_xlabel("")
ax.set_ylabel("")

# 14. Set map extent
ax.set_xlim(
    68,
    98
)

ax.set_ylim(
    5,
    37
)

# 15. Adjust layout
plt.tight_layout()

# 16. Display map
plt.show()