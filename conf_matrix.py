import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# conf_matrix = np.array([[179, 15, 13, 13, 55, 22, 61, 20,  9, 21,  6, 32,  9, 31, 18, 51, 10, 12,  19, 23],
#  [2, 108,  2,  2,  1,  2, 10,  7,  1,  2,  0,  1,  2,  5,  5,  3,  4,  2, 1,  2],
#  [0,  3, 114,  9,  5,  7,  0,  7,  3,  6,  2,  8,  2,  2, 14,  1,  3,  8, 0,  1],
#  [6,  9,  5, 137,  6,  2,  8,  9, 13,  3,  4,  7,  1,  6, 25,  9, 11, 19, 2, 17],
#  [ 23,  2, 22,  3, 291, 15,  1, 11,  7, 20,  4, 82,  4, 10, 48, 12,  6, 14,  18,  7],
#  [2,  5, 13, 15,  9, 110,  0, 11,  4, 10,  4, 13,  2,  8, 46, 11, 10, 13, 3,  6],
#  [ 20, 53, 12, 16,  2, 11, 136,  6,  4,  7,  8, 11, 15, 64, 10, 11, 14,  7, 3, 10],
#  [0,  2,  2,  5,  4,  2,  0, 89,  1,  3,  2,  2,  3,  5,  3,  1,  2, 12, 0,  7],
#  [ 15, 13, 18, 35,  4,  4, 16,  4, 876,  7, 16,  8, 12, 31, 19,  5, 16, 12, 4, 11],
#  [2, 11, 12, 12, 13, 14,  7, 13,  7, 118,  0,  8,  2,  6, 15, 13, 10, 19, 3, 12],
#  [2,  1,  5,  4,  1,  1,  1,  4,  4,  0, 109,  7,  0, 10,  2,  1,  0,  2, 2,  2],
#  [19,  6, 34, 23, 77, 19,  0, 13,  9, 10,  5, 142,  4,  9, 44, 19,  9, 25, 8,  6],
#  [0,  2,  0,  3,  0,  2,  2,  3,  1,  3,  0,  0, 139,  9,  4,  2,  1,  6, 0,  2],
#  [ 7, 12,  1, 14,  3,  4, 12,  8,  3,  2, 12,  2,  7, 145,  9,  6,  3,  2, 3,  3],
#  [0,  1,  3,  0,  2,  8,  0,  1,  1,  1,  1,  1,  1,  1, 124,  2,  3,  4, 0,  0],
#  [ 46, 20, 14, 11, 29, 21, 15, 14,  7, 17,  3, 26, 17, 22, 20, 132, 19, 14,  14, 16],
#  [ 3,  4,  6,  6,  2,  9,  4,  2,  3,  2,  1,  5,  7,  1,  7,  3, 67,  7, 2,  9],
#  [1,  4,  8,  9,  5,  8,  3,  9,  7,  9,  0,  9,  8,  6, 23,  3,  5, 135, 0,  8],
#  [ 4,  5,  9,  5, 16,  5,  0,  2,  3,  7,  0,  3,  1,  3,  1,  6,  2,  6, 257,  8],
#  [1,  4,  1,  2,  0,  1,  1,  8,  2,  0,  2,  0,  1,  4,  0,  2,  4,  2, 3, 107]])

# conf_matrix = 100 * conf_matrix / conf_matrix.sum(axis=1)[:, None]

conf_matrix = 100*np.load('conf_matrix_norm.npy')

species_mapping = {
"Acer pseudoplatanus": 0,
"Alnus glutinosa": 1,
"Pyrus calleryana": 2,
"Prunus avium": 3,
"Tilia x euchlora": 4,
"Acer campestre": 5,
"Fraxinus excelsior": 6,
"Pinus nigra": 7,
"Platanus x acerifolia": 8,
"Carpinus betulus": 9,
"Styphnolobium japonicum": 10,
"Tilia cordata": 11,
"Populus nigra": 12,
"Robinia pseudoacacia": 13,
"Alnus x spaethii": 14,
"Acer platanoides": 15,
"Quercus robur": 16,
"Betula pendula": 17,
"Aesculus hippocastanum": 18,
"Taxus baccata": 19
}

import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

# Les bornes que vous voulez différencier
# bounds = [0, 5, 10, 20, 45, 60, 75, 100]
bounds = [0, 5, 10, 30, 60, 70, 80, 90, 100]

cmap = plt.get_cmap('Blues')
newcolors = cmap(np.linspace(0, 1, 256))
newcolors[0] = np.array([1, 1, 1, 1])
cmap = ListedColormap(newcolors)
# cmap = plt.get_cmap('rainbow')
# Normalisation des couleurs pour qu'elles correspondent aux bornes
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(15, 10))
sns.set(font_scale=1) # font size
# ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(species_mapping.keys()), yticklabels=list(species_mapping.keys()))
# ax = sns.heatmap(conf_matrix, annot=True, fmt=".0f", cmap="tab20", xticklabels=list(species_mapping.keys()), yticklabels=list(species_mapping.keys()), vmax=100)
ax = sns.heatmap(conf_matrix, annot=True, fmt=".0f", cmap=cmap, norm=norm, xticklabels=list(species_mapping.keys()), yticklabels=list(species_mapping.keys()), vmax=100)
plt.title('Confusion Matrix', fontsize=18)
plt.ylabel('Actual Labels', fontsize=18)
plt.xlabel('Predicted Labels', fontsize=18)

ax.tick_params(axis='both', labelsize=12)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.savefig("./conf_matrix10.png")


# # Creation and saving of the figure for the confusion matrix
# plt.figure(figsize=(10, 8))
# sns.set(font_scale=1.2)  # font size
# ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(self.species_mapping.keys()), yticklabels=list(self.species_mapping.keys()))
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Labels')
# plt.xlabel('Predicted Labels')

# # Rotation of the labels on the x-axis for better readability
# plt.xticks(rotation=45, ha="right")
# plt.yticks(rotation=0)
# plt.tight_layout()