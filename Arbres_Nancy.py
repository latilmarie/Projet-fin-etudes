"""
Arbres_Nancy.py
---------------
This script filters the trees in Nancy to keep only the trees present in Strasbourg with the same nomenclature.
"""

import geopandas as gpd

nancy_shp = '/home/marie/Téléchargements/Arbres_nancy/arbres.shp'
stras_shp = '/home/marie/Documents/data/PA15_tamp.gpkg'

save = False

nancy = gpd.read_file(nancy_shp)
stras = gpd.read_file(stras_shp, engine='pyogrio')
print("Number of trees in Nancy before filtering:", len(nancy))
print("Number of trees in Strasbourg:", len(stras))

# Filter Nancy trees by gender present in Strasbourg, column 'genre'
genres_nancy = nancy['GENRE']
genres_stras = set(stras['genre'])
filtered_nancy = nancy[genres_nancy.isin(genres_stras)]
print("Genders in Strasbourg:", genres_stras)
print("\nGenders in Nancy after filtering only the species in Strasbourg:", filtered_nancy['GENRE'].unique())
print("Number of trees after filtering per gender:", len(filtered_nancy))

# Replace 'x_europaea' by 'x_euchlora' in the 'ESPECE' column
filtered_nancy['ESPECE'] = filtered_nancy['ESPECE'].replace('x_europaea', 'x_euchlora')

# New column 'esse_tri': Concatenate the columns "GENRE" and "ESPECE" with a space in the middle and replace underscores with spaces
filtered_nancy['esse_tri'] = (filtered_nancy['GENRE'] + ' ' + filtered_nancy['ESPECE'].str.replace('_', ' '))
print("Species in Nancy with the new column 'esse_tri' :", filtered_nancy['esse_tri'].unique())

# Filter Nancy trees by species present in Strasbourg
species_nancy0 = filtered_nancy['esse_tri']
species_stras = set(stras['esse_tri'])
species_nancy = filtered_nancy[species_nancy0.isin(species_stras)]
print("Species in Strasbourg:", species_stras)
print("\nSpecies in Nancy after filtering:", species_nancy['esse_tri'].unique())
print("Number of trees after filtering by species:", len(species_nancy))

# Exporter le GeoDataFrame filtré en shapefile
if save:
    output_shapefile_path = '/home/marie/Téléchargements/Arbres_nancy/arbres_filtres.shp'
    species_nancy.to_file(output_shapefile_path)
    print(f"Filtered shapefile exported to: {output_shapefile_path}")
