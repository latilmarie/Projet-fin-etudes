import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

import params
from models import DualTransformer
import test
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  


# Model parameters
d_model = 64 # 64 // 128 // 256  # Lattent dim
q = 8  # 64/8 = 8 // 128/8 = 16 // 256/8 = 32 Query size
v = 8  # 64/8 = 8 // 128/8 = 16 // 256/8 = 32 Value size
h = 4  # 8 // 4 Number of heads
N = 4  # 4 // 6 Number of encoder and decoder to stack
attention_size = 12 # 12 // 8 // 16 # Attention window size
dropout = 0.2 # 0.1 // 0.2  # Dropout rate
pe = 'original'  # Positional encoding
chunk_mode = None

d_input = 4  # From dataset, 10 or 4
d_output = 20  # From dataset, number of classes
model = DualTransformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
        dropout=dropout, chunk_mode=chunk_mode, pe=pe)
model.to(device)
            
      


# model_paths = []
# for n in range(params.n_splits):
#     model_paths.append(os.path.join(params.model_path, f'model_{n}.pth'))
    
# Load the models from memory
name_scenar=f"{params.name_model}12-k{params.k}-{params.num_classes}species-{params.use_data}-{params.n_splits}times_old"
path_output = f'/home2020/home/geo/mlatil/Tree-species-deep/results/{name_scenar}'
model_paths = os.path.join(params.model_path, f'model_0.pth')
# Scenar

### Chemins des fichiers et dossiers
# spectral_signature_dir = './spectral_signatures'
shapefile_path = "/home2020/home/geo/mlatil/data/PA15_tamp.gpkg"
path_output = f'/home2020/home/geo/mlatil/Tree-species-deep/results/{name_scenar}'

output_filename = f"SITS_UrbanTrees_{name_scenar}.out"
# print("output filename:", output_filename)

gradcam_boxplot_dir = f"/home2020/home/geo/mlatil/Tree-species-deep/results/GradCAM_boxplots/{name_scenar}"
model_path = f'/home2020/home/geo/mlatil/Tree-species-deep/models/model-{name_scenar}'


model.load_state_dict(torch.load(model_paths, map_location=device))
model.to(device)
model.eval()

# models = [test.load_model(path, model, device) for path in model_paths]
# models = test.load_model(model_paths, model, device)

# print(models)
# print(models.layers_encoding[0].attention_map[0].cpu().detach().numpy().shape)
    
# for n in range(params.n_splits):
#     # Select first encoding layer
#     encoder = models[n].layers_encoding[0]

#     # Get the first attention map
#     attn_map = encoder.attention_map[0].cpu()

#     # Plot
#     plt.figure(figsize=(20, 20))
#     sns.heatmap(attn_map)
#     plt.savefig(os.path.join(params.path_output, f'attention_map_{n}.png'))
    
# model = torch.load(model_paths, map_location=device)




# Créer une figure et une grille de sous-parcelles (4x4)
fig, axs = plt.subplots(N, h, figsize=(15, 15))

# Parcourir les couches et les têtes pour afficher chaque carte d'attention
for i in range(N):
    for j in range(h):
        attention_maps = model.layers_encoding[i].attention_map[j].cpu().detach().numpy()
        # Afficher la carte d'attention
        ax = axs[i, j]
        ax.imshow(attention_maps[i, j], cmap='viridis')
        ax.set_title(f'Layer {i+1}, Head {j+1}')
        ax.axis('off')  # Désactiver les axes pour plus de clarté

# Ajuster l'espacement entre les sous-parcelles
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(params.path_output, f'attention_maps.png'))