"""
Number_trees_out.py
-------------------
This script computes the number of no data trees for each PlanetScope raster in 2022 in Nancy.
It counts the frequency of missing data for each tree and plot the results.
"""

import geopandas as gpd
import os
from matplotlib import pyplot as plt
import rasterio
import numpy as np
import glob
from rasterio.plot import show, adjust_band
import matplotlib.cm as cm
import re
from matplotlib.patches import FancyArrow
# import geodatasets
from geo_northarrow import add_north_arrow
from matplotlib_scalebar.scalebar import ScaleBar

compute = False
save = True

path_raster2 = '/media/marie/LaCie/NANCY_PS_2022/2022/Archives/ORTHO_NANCY/220708_ORTHO_NANCY/files/PSOrthoTile/PS-08072022_masked.tif'
path_raster = '/media/marie/LaCie/NANCY_PS_2022/2022/Archives/ORTHO_NANCY/'

if compute:
    path_trees = '/home/marie/Téléchargements/Arbres_nancy/arbres_filtres.shp'
else:
    path_trees = '/home/marie/Téléchargements/Arbres_nancy/arbres_filtres_count2.shp'

points_gdf = gpd.read_file(path_trees)
points_gdf['count_outside_mask'] = 0 if compute else points_gdf['count_outs']

# Function to get the value of a pixel
def is_outside_mask(point, raster, transform):
    x, y = point.x, point.y
    row, col = ~transform * (x, y)
    row, col = int(row), int(col)
    return raster[col, row] == 0

# Function to update the count of points outside the mask
def update_count_outside_mask(raster_path, points_gdf):
    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        raster_transform = src.transform
        raster_crs = src.crs
        points_gdf = points_gdf.to_crs(raster_crs)
        points_gdf['outside_mask'] = points_gdf['geometry'].apply(lambda point: is_outside_mask(point, raster, raster_transform))
        points_gdf['count_outside_mask'] += points_gdf['outside_mask'].astype(int)
    return points_gdf

# Function to linearly stretch the contrast and adjust the brightness of a band
def linear_stretch_brightness(band, brightness_factor):
    p2, p98 = np.percentile(band, (2, 98))
    image = np.clip((band - p2) / (p98 - p2), 0, 1)
    return np.clip(image + brightness_factor, 0, 1)

# ______________________________________________________________________________________
# To compute the number of trees outside the mask for each raster
if compute:
    num_points = points_gdf.shape[0]
    print(f"Total number of points in the shapefile : {num_points}")
    print(f"Classes distribution per species : {points_gdf['esse_tri'].value_counts()}")
    raster_files = sorted(glob.glob(os.path.join(path_raster, "*/files/PSOrthoTile/*_masked.tif")))
    for raster_file in raster_files:
        points_gdf_before = points_gdf['count_outside_mask'].sum()
        points_gdf = update_count_outside_mask(raster_file, points_gdf)
        print(f"Total number of points outside at date: {re.search(r'(\d{8})', raster_file).group(1)} : {points_gdf['count_outside_mask'].sum() - points_gdf_before}")

outside_points = points_gdf[points_gdf['count_outside_mask'] > 0] # filter points that are outside the mask at least once
counts = outside_points['count_outside_mask'] # get the values of count_outside_mask

# Plot the raster and the points with the counter
fig, ax = plt.subplots(figsize=(10, 10))
cmap = cm.get_cmap('plasma')
norm = plt.Normalize(vmin=0, vmax=counts.max())
colors = cmap(norm(counts))
brightness_factor = 0.1

with rasterio.open(path_raster2) as src:
    red = adjust_band(src.read(1))
    green = adjust_band(src.read(2))
    blue = adjust_band(src.read(3))

    red = linear_stretch_brightness(red, brightness_factor)
    green = linear_stretch_brightness(green, brightness_factor)
    blue = linear_stretch_brightness(blue, brightness_factor)

    rgb = np.dstack((red, green, blue))
    rgb[rgb == 0.1] = 1
    ax.imshow(rgb, extent=(src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]))

scatter = ax.scatter(outside_points.geometry.x, outside_points.geometry.y, s=25, color=colors, marker='o', linewidths=0.6, edgecolors='black', label='Missing data tree points')
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.7)
cbar.set_label('Count of missing tree data')
ax.legend()
# ax.set_title('Number of points outside the mask PS 2022')
ax.grid(False)
ax.axis('off')
add_north_arrow(ax, scale=.5, xlim_pos=.1, ylim_pos=.965, color='#000', text_scaler=2, text_yT=-1.25)
ax.add_artist(ScaleBar(1, location='lower right'))
fig.tight_layout()
if save:
    plt.savefig('/home/marie/Documents/Processing_nancy/Number_trees_out/Number_of_trees_out2.png', dpi=300)
plt.show()

# Save the updated shapefile
if compute and save:
    output_shapefile_path = '/home/marie/Téléchargements/Arbres_nancy/arbres_filtres_count2.shp'
    points_gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
    print(f"Shapefile updated to: {output_shapefile_path}")
