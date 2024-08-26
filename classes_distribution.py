# second_programme.py
import torch
# from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import os
import numpy as np
import sys
import geopandas as gpd
from sklearn.model_selection import train_test_split

sys.path.append('/home2020/home/geo/mlatil/Tree-species-deep/')
import params
from utils import filter_species

shapefile_path = "/home2020/home/geo/mlatil/data/PA15_tamp.gpkg"
shapefile_path_nancy = "/home2020/home/geo/mlatil/data/PA15_tamp_Nancy2.gpkg"

# Strasbourg
data = gpd.read_file(shapefile_path, engine="pyogrio")
data = filter_species(data, params.libelle, params.selected_species_20)

labels = data[params.libelle]
train_data, test_data = train_test_split(data, test_size=0.15, random_state=11, stratify=labels)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=11, stratify=train_data[params.libelle]) #0.25

bars = data[params.libelle].value_counts()
train_bars = train_data[params.libelle].value_counts()
val_bars = val_data[params.libelle].value_counts()
test_bars = test_data[params.libelle].value_counts()


# Nancy
data = gpd.read_file(shapefile_path_nancy, engine="pyogrio")
data = filter_species(data, params.libelle, params.selected_species_19)

labels = data[params.libelle]
train_data, test_data = train_test_split(data, test_size=0.15, random_state=11, stratify=labels)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=11, stratify=train_data[params.libelle]) #0.25

bars = data[params.libelle].value_counts()
train_bars_2 = train_data[params.libelle].value_counts()
val_bars_2 = val_data[params.libelle].value_counts()
test_bars_2 = test_data[params.libelle].value_counts()



train_bars_2['Styphnolobium japonicum'] = 0
val_bars_2['Styphnolobium japonicum'] = 0
test_bars_2['Styphnolobium japonicum'] = 0

train_bars_2 = train_bars_2.reindex(train_bars.index, fill_value=0)
val_bars_2 = val_bars_2.reindex(val_bars.index, fill_value=0)
test_bars_2 = test_bars_2.reindex(test_bars.index, fill_value=0)

# Bar plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["top"].set_visible(False)
bar_width = 0.5
indices = np.arange(0, len(train_bars_2.index)*2, 2)

train_color = 'darkred'
val_color = 'tab:blue'
test_color = 'darkgreen'
city_color = 'white'

# p1 = ax.bar(indices-0.5, bars, bar_width, label='Total', color='darkblue', alpha=0.6)
p2 = ax.bar(indices-0.75, train_bars, bar_width, label='Train', color=train_color, alpha=0.6, edgecolor='black')
p3 = ax.bar(indices-0.75, val_bars, bar_width, label='Validation', color=val_color, alpha=0.6, bottom=train_bars, edgecolor='black')
p4 = ax.bar(indices-0.75, test_bars, bar_width, label='Test', color=test_color, alpha=0.6, bottom=train_bars+val_bars, edgecolor='black')

p5 = ax.bar(indices, train_bars_2, bar_width, color=train_color, alpha=0.6, hatch='//', edgecolor='black')
p6 = ax.bar(indices, val_bars_2, bar_width, color=val_color, alpha=0.6, bottom=train_bars_2, hatch='//', edgecolor='black')
p7 = ax.bar(indices, test_bars_2, bar_width, color=test_color, alpha=0.6, bottom=train_bars_2+val_bars_2, hatch='//', edgecolor='black')

strasbourg_patch = Rectangle((0,0),1,1, facecolor=city_color, edgecolor='black', label='Strasbourg')
nancy_patch = Rectangle((0,0),1,1, facecolor=city_color, edgecolor='black', hatch='//', label='Nancy')
ax.legend(handles=[p2, p3, p4, strasbourg_patch, nancy_patch], loc='best', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('Species', fontsize=15)
ax.set_ylabel('Number of trees', fontsize=15)
# ax.set_title('Trees distribution per species between train, validation and test sets')
ax.set_xticks(indices-0.375)
ax.set_xticklabels(train_bars.index, rotation=45, ha='right', fontsize=12)

# ax.legend()
plt.tight_layout()
plt.show()
plt.savefig(f"./classes_distribution_global18.png")