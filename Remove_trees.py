import geopandas as gpd
import os
from matplotlib import pyplot as plt
import rasterio
import numpy as np
from rasterio.features import rasterize
import fiona
import glob
from shapely.geometry import mapping
from shapely.geometry import shape, Point
from rasterio.plot import show
import matplotlib.cm as cm
import re
import pandas as pd

# Patrimoine arboré Nancy (shapefile)
path_trees = '/home/marie/Téléchargements/Arbres_nancy/arbres_filtres.shp'

# Remove trees cut on Planet (on the right)
path_raster_ps = '/media/marie/LaCie/NANCY_PS_2022/2022/Archives/ORTHO_NANCY/220831_ORTHO_NANCY/files/PSOrthoTile/PS-31082022_masked.tif'

# Remove trees cut on S2 (on the bottom)
path_raster_s2 = '/media/marie/LaCie/NANCY_S2_2022/Cube/B8A/T31UGQ_20220112T104319_B8A_20m.tif'

# Fonction pour obtenir la valeur d'un pixel
def is_outside_mask(point, raster, transform):
    x, y = point.x, point.y
    row, col = ~transform * (x, y)
    row, col = int(row), int(col)
    
    if (row < 0 or row >= raster.shape[0] or col < 0 or col >= raster.shape[1]):
        return True
    else:
        return raster[col, row] == 0
    # return raster[col, row] == 0

# Fonction pour supprimer les points en dehors du raster
def remove_points_outside_mask(raster_path, points_gdf):
    with rasterio.open(raster_path) as src:
        raster = src.read(1)  # Lire la première bande
        raster_transform = src.transform  # Transformation du raster
        raster_crs = src.crs  # Système de coordonnées du raster
        points_gdf = points_gdf.to_crs(raster_crs)  # Reprojeter les points dans le système de coordonnées du raster
        
        initial_count = len(points_gdf)
        
        # points_gdf = points_gdf[~points_gdf['geometry'].apply(lambda point: is_outside_mask(point, raster, raster_transform))]
        
        points_outside_mask = points_gdf['geometry'].apply(lambda point: is_outside_mask(point, raster, raster_transform))
        removed_points_gdf = points_gdf[points_outside_mask]
        
        points_gdf = points_gdf[~points_outside_mask]
        
        final_count = len(points_gdf)
        points_removed = initial_count - final_count
        
    return points_gdf, removed_points_gdf, points_removed

# ______________________________________________________________________________________

# Charger le shapefile
points_gdf = gpd.read_file(path_trees)
# total_species_counts = points_gdf['esse_tri'].value_counts()
total_species_counts = points_gdf['esse_tri'].value_counts().sort_index()

# Afficher le nombre de points dans le shapefile
print(f"Nombre total de points dans le shapefile : {len(points_gdf)}")

print(f"Répartition des classes par espèce : {total_species_counts}")

total_points_removed = 0
species_removal_counts = pd.Series(dtype=int)

# Processus pour Planet
points_gdf, removed_points_gdf, points_removed = remove_points_outside_mask(path_raster_ps, points_gdf)
total_points_removed += points_removed
print(f"Nombre de points supprimés pour le raster Planet du {re.search(r'(\d{8})', path_raster_ps).group(1)} : {points_removed}")


# Calculer les nombres d'arbres supprimés par respèce et pourcentage
removed_species_counts = removed_points_gdf['esse_tri'].value_counts()
species_removal_counts = species_removal_counts.add(removed_species_counts, fill_value=0)
species_removal_counts = species_removal_counts.reindex(total_species_counts.index, fill_value=0)
percentage_removed = (species_removal_counts / total_species_counts) * 100
print("Nombre d'arbres supprimés par espèce :", species_removal_counts)
print("Pourcentage d'arbres supprimés par espèce :", percentage_removed)


species_removal_counts = pd.Series(dtype=int)
    
# Processus pour Sentinel-2
points_gdf, removed_points_gdf, points_removed = remove_points_outside_mask(path_raster_s2, points_gdf)
total_points_removed += points_removed
print(f"Nombre de points supprimés pour le raster S2 du {re.search(r'(\d{8})', path_raster_s2).group(1)} : {points_removed}")

# Calculer les nombres d'arbres supprimés par respèce et pourcentage
removed_species_counts = removed_points_gdf['esse_tri'].value_counts()
species_removal_counts = species_removal_counts.add(removed_species_counts, fill_value=0)
species_removal_counts = species_removal_counts.reindex(total_species_counts.index, fill_value=0)
percentage_removed = (species_removal_counts / total_species_counts) * 100
print("Nombre d'arbres supprimés par espèce :", species_removal_counts)
print("Pourcentage d'arbres supprimés par espèce :", percentage_removed)


# Afficher le nombre total de points supprimés
print(f"Nombre total de points supprimés : {total_points_removed}")
print(f"Nombre total de points restants dans le shapefile : {len(points_gdf)}")


# Enregistrer le shapefile mis à jour
output_shapefile_path = '/home/marie/Téléchargements/Arbres_nancy/arbres_filtres_remove_trees.gpkg'
# points_gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
points_gdf.to_file(output_shapefile_path, driver='GPKG', layer='arbres_filtres_remove_trees')
print(f"Le shapefile mis à jour a été enregistré à {output_shapefile_path}")
