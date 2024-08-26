# finetune.py
# Standard library imports
import os
import shutil

# Data manipulation and numerical computing
import numpy as np
import pandas as pd

# Machine learning and data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

# Deep Learning with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, StratifiedKFold

# Geospatial data processing and visualization
import geopandas as gpd
import json

# Other imports
from datetime import datetime
import time
import re
from codecarbon import OfflineEmissionsTracker

# Other project files
import params
from utils import CustomDataset, CustomDualDataset, extract_features_2D_tensor, global_normalize, filter_species, get_attention_maps_by_class
import test
from train import TrainModel
from models import InceptionTime, DualInceptionTime, HybridInceptionTime, DualHybridInceptionTime, Transformer, DualTransformer, TSTransformerEncoderClassiregressor, TSTransformerEncoderClassiregressorDual, LITE, DualLITE


#CARBON Tracker
tracker = OfflineEmissionsTracker(country_iso_code="FRA")
tracker.start()

start = time.time()

# Creation of the output folders and deletion of the previous ones if it exists
paths = [params.path_output, params.model_path]
for p in paths:
    if os.path.exists(p):
        shutil.rmtree(p)
        os.makedirs(p)
    else:
        os.makedirs(p)

# General information
print("[INFO] General information parameters",
        "\n Scenar:", params.name_scenar,
        "\n Model:", params.name_model, 
        "\n Number of classes:", params.num_classes, 
        "\n Number of splits:", params.n_splits,
        "\n Batch size:", params.batch_size,
        "\n Number of epochs:", params.num_epochs,
        "\n Learning rate:", params.learning_rate,
        "\n Data used:", params.use_data,
        "\n Use t-SNE:", params.tsne,
        "\n Kernel sizes S2:", params.kernel_sizes_s2 if re.search(r'S2', params.use_data) else None,
        "\n Kernel sizes Planet:", params.kernel_sizes_planet if re.search(r'Planet', params.use_data) else None,
        "\n Interpolate:", params.interpolate,
        "\n"
        )

# Setting the device to GPU if available
print("[INFO] Initializing...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device set to {device}")

# kfold = KFold(n_splits=params.n_splits, shuffle=True, random_state=11)
stratified_kfold = StratifiedKFold(n_splits=params.n_splits, shuffle=True, random_state=11)

# Read the Shapefile using GeoPandas and filter the data according to the desired species (10 or 20)
print("[INFO] Shapefile loading...")
start1 = time.time()
data = gpd.read_file(params.shapefile_path, engine="pyogrio")
print("[INFO] Shapefile loaded in", (time.time()-start1), "seconds.")
print(data.shape)

data = filter_species(data, params.libelle, params.selected_species)

# Labels
labels = data[params.libelle]

# Splitting the data
print("[INFO] Splitting data...")
train_data, test_data = train_test_split(data, test_size=0.15, random_state=11, stratify=labels)
#train_data, test_data = train_test_split(data, test_size=0.15)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=11, stratify=train_data[params.libelle]) #0.25
print("[INFO] Data split into training, validation, and test sets.")

# devide the train set by 4
# train_data, train_data_rest = train_test_split(train_data, test_size=0.75, random_state=11, stratify=train_data[params.libelle]) #0.25

# Check the distribution of classes in each set
print("Distribution of classes in the original dataset:", data[params.libelle].value_counts())
print("Distribution of classes in the training dataset:", train_data[params.libelle].value_counts())
print("Distribution of classes in the validation dataset:", val_data[params.libelle].value_counts())
print("Distribution of classes in the test dataset:", test_data[params.libelle].value_counts())

# Number of trees for each set
print("Number of trees in total:", data.shape[0])
print("Number of trees in the train set:", train_data.shape[0])
print("Number of trees in the validation set:", val_data.shape[0])
print("Number of trees in the test set:", test_data.shape[0])
# print("Number of trees in the train set rest:", train_data_rest.shape[0])
    
# List of species present in the Shapefile
especes = data[params.libelle].unique() # 19

# Encode the species labels into integers
species_mapping = {species: i for i, species in enumerate(especes)} # number of trees for each species
print("Species mapping:", species_mapping)
inverse_species_mapping = {v: j for j, v in species_mapping.items()}
train_labels = np.array([species_mapping[species] for species in train_data[params.libelle]])
val_labels = np.array([species_mapping[species] for species in val_data[params.libelle]])
test_labels = np.array([species_mapping[species] for species in test_data[params.libelle]])
print("[INFO] Data filtered by selected species. \n")

# Binairize and convert labels to PyTorch tensors
label_binarizer = LabelBinarizer() # binarize en one-hot encoder
train_labels_tensor = torch.tensor(label_binarizer.fit_transform(train_labels)).float() # size n, 20
val_labels_tensor = torch.tensor(label_binarizer.transform(val_labels)).float()
test_labels_tensor = torch.tensor(label_binarizer.transform(test_labels)).float()
# train_val_test_labels_tensor = torch.cat((train_labels_tensor, val_labels_tensor, test_labels_tensor), dim=0)
print("Label size:", train_labels_tensor.shape, val_labels_tensor.shape, test_labels_tensor.shape)

# Values from Strasbourg dataset
min_s2_stras = torch.tensor(3.)
max_s2_stras = torch.tensor(65518.)
min_planet_stras = torch.tensor(1.)
max_planet_stras = torch.tensor(58846.)

mean_s2_stras = torch.tensor(9728.5488)
std_s2_stras = torch.tensor(3779.1118)
mean_planet_stras = torch.tensor(1086.4323)
std_planet_stras = torch.tensor(895.3278)

# Load Data and extract features from data
if not params.use_multisensors:
    print(f"[INFO] Loading {params.use_data} data...")
    train_tensor = extract_features_2D_tensor(train_data, prefix=params.use_data)
    print("Dimensions of the training tensor:", train_tensor.shape)
    val_tensor = extract_features_2D_tensor(val_data, prefix=params.use_data)
    print("Dimensions of the validation tensor:", val_tensor.shape)
    test_tensor = extract_features_2D_tensor(test_data, prefix=params.use_data)
    print("Dimensions of the test tensor:", test_tensor.shape)
    
    # Normalization
    train_features_tensor_norm, val_features_tensor_norm, test_features_tensor_norm = global_normalize(train_tensor, val_tensor, test_tensor, params.type_norm)
    print("[INFO] Features normalized. \n")
    print("[INFO] Time: ", (time.time()-start)/60, "minutes.")
        
else:
    # S2 and Planet - Create the 2D tensor for training, validation, and test sets
    print("[INFO] Loading Sentinel-2 and Planet data...")
    train_tensor_s2, train_tensor_planet = extract_features_2D_tensor(train_data)
    val_tensor_s2, val_tensor_planet = extract_features_2D_tensor(val_data)
    test_tensor_s2, test_tensor_planet = extract_features_2D_tensor(test_data)
    print("S2 - Dimensions of the training tensor:", train_tensor_s2.shape)
    print("S2 - Dimensions of the validation tensor:", val_tensor_s2.shape)
    print("S2 - Dimensions of the test tensor:", test_tensor_s2.shape)
    print("Planet - Dimensions of the training tensor:", train_tensor_planet.shape)
    print("Planet - Dimensions of the validation tensor:", val_tensor_planet.shape)
    print("Planet - Dimensions of the test tensor:", test_tensor_planet.shape)

    # Normalization
    print("[INFO] Normalizing features...")
    print("Normalizing S2 features...")
    train_features_tensor_norm_s2, val_features_tensor_norm_s2, test_features_tensor_norm_s2 = global_normalize(train_tensor_s2, val_tensor_s2, test_tensor_s2, params.type_norm)
    print("Normalizing Planet features...")
    train_features_tensor_norm_planet, val_features_tensor_norm_planet, test_features_tensor_norm_planet = global_normalize(train_tensor_planet, val_tensor_planet, test_tensor_planet, params.type_norm)
    print("[INFO] Features normalized. \n")
    print("[INFO] Time: ", (time.time()-start)/60, "minutes.")
    
print("[INFO] Features normalized. \n")
print("[INFO] Time: ", (time.time()-start)/60, "minutes.")

# Creation of the model's architecture to apply and train with
if params.name_model == 'InceptionTime':
    model = InceptionTime(in_channels=params.bands_depth, 
                            number_classes=params.num_classes, 
                            use_residual=True, 
                            activation=nn.ReLU())
    model.to(device)
elif params.name_model == 'DualInceptionTime':
    model = DualInceptionTime(in_channels_s2=params.bands_depth_S2, 
                                in_channels_planet=params.bands_depth_Planet, 
                                kernel_sizes_s2= params.kernel_sizes_s2, 
                                kernel_sizes_planet = params.kernel_sizes_planet, 
                                number_classes=params.num_classes)
    model.to(device)
elif params.name_model == "HInceptionTime":
    model = HybridInceptionTime(in_channels=params.bands_depth, 
                                kernel_sizes= params.kernel_sizes, 
                                number_classes=params.num_classes)
    model.to(device)
elif params.name_model == "HDualInceptionTime":
    model = DualHybridInceptionTime(in_channels_s2=params.bands_depth_S2, 
                                    in_channels_planet=params.bands_depth_Planet, 
                                    kernel_sizes_s2= params.kernel_sizes_s2, 
                                    kernel_sizes_planet = params.kernel_sizes_planet, 
                                    number_classes=params.num_classes)
    model.to(device)
elif params.name_model == "Transformer":            
    ## Model 1
    if params.transf == 1:
        model = Transformer(d_input = params.bands_depth, 
                            d_model = params.d_model, 
                            d_output = params.num_classes, 
                            q=params.q, v=params.v, h=params.h, N=params.N, 
                            attention_size=params.attention_size,
                            dropout=params.dropout)
        model.to(device)
    ## Model 2
    elif params.transf == 2:
        model = TSTransformerEncoderClassiregressor(feat_dim=params.bands_depth, 
                                                    d_model=params.d_model,
                                                    n_heads=params.h,
                                                    num_layers=params.N,
                                                    dim_feedforward=params.dim_feedforward,
                                                    num_classes=params.num_classes,
                                                    dropout=params.dropout,
                                                    activation=nn.ReLU(),
                                                    norm=params.normalization_layer)
        model.to(device)            
elif params.name_model == "DualTransformer":
    ## Model 1
    if params.transf == 1:
        model = DualTransformer(d_input=params.bands_depth_Planet,
                                d_model=params.d_model, 
                                d_output=params.num_classes, 
                                q=params.q, v=params.v, h=params.h, N=params.N, 
                                attention_size=params.attention_size,
                                dropout=params.dropout)
        model.to(device)            
    ## Model 2
    elif params.transf == 2:
        model = TSTransformerEncoderClassiregressorDual(feat_dim_s2=params.bands_depth_S2, 
                                                        feat_dim_planet=params.bands_depth_Planet, 
                                                        d_model=params.d_model,
                                                        n_heads=params.h,
                                                        num_layers=params.N, 
                                                        dim_feedforward=params.dim_feedforward,
                                                        num_classes=params.num_classes,
                                                        dropout=params.dropout, 
                                                        activation=nn.ReLU(),
                                                        norm=params.normalization_layer)
        model.to(device)
elif params.name_model == "LITE":
    model = LITE(in_channels=params.bands_depth, 
                    kernel_sizes=params.kernel_sizes, 
                    number_classes=params.num_classes)
    model.to(device)
elif params.name_model == "DualLITE":
    model = DualLITE(in_channels_s2=params.bands_depth_S2, 
                        in_channels_planet=params.bands_depth_Planet, 
                        kernel_sizes_s2= params.kernel_sizes_s2, 
                        kernel_sizes_planet=params.kernel_sizes_planet, 
                        number_classes=params.num_classes)
    model.to(device)

model_paths = []
for n in range(params.n_splits):
    model_paths.append(os.path.join(params.model_path_stras, f'model_{n}.pth'))

# Load the models from memory
models = [test.load_model(path, model, device) for path in model_paths]

# Compute the class frequencies
class_counts = np.bincount(train_labels)

# Compute the class weights for each class (Inverse of the frequencies)
# classes with fewer samples will have higher weights: "pay more attention to the under-represented classes"
class_weights = 10000*(1. / class_counts)
# class_weights = 1000000*(1. / class_counts)

# Convertir en Tensor de PyTorch
class_weights = torch.FloatTensor(class_weights).to(device)

### Start of the training process
print(f"[INFO] Starting training {params.n_splits} runs ... \n")
# Creation of a dictionnary to store the results of training for each run (accuracy, precision, recall, f1-score)
results = {
    'accuracy': [],
    'precision': {class_name: [] for class_name in species_mapping.keys()},
    'recall': {class_name: [] for class_name in species_mapping.keys()},
    'f1-score': {class_name: [] for class_name in species_mapping.keys()},
}
        
if not params.use_multisensors:
    train_dataset = CustomDataset(train_features_tensor_norm, train_labels_tensor)
    val_dataset = CustomDataset(val_features_tensor_norm, val_labels_tensor)
    test_dataset = CustomDataset(test_features_tensor_norm, test_labels_tensor)
else:
    train_dataset = CustomDualDataset(train_features_tensor_norm_s2, train_features_tensor_norm_planet, train_labels_tensor)
    val_dataset = CustomDualDataset(val_features_tensor_norm_s2, val_features_tensor_norm_planet, val_labels_tensor)
    test_dataset = CustomDualDataset(test_features_tensor_norm_s2, test_features_tensor_norm_planet, test_labels_tensor)

for n in range(len(models)):
    model = models[n]
    # Creation of the training model
    test_model = TrainModel(model, train_dataset, val_dataset, test_dataset, class_weights, species_mapping)

    # Training
    print(f"[INFO] Starting training for fold {n} ...")
    report = test_model.run_n_times(n)
    results['accuracy'].append(report['accuracy'])

    # Precision, recall, F1-score appened to their respective lists in the results dictionnary (if the class is in the species_mapping dictionnary)
    for class_name, metrics in report.items():
        if class_name in species_mapping.keys():  # Pour éviter les clés comme 'accuracy' et 'macro avg'
            results['precision'][class_name].append(metrics['precision'])
            results['recall'][class_name].append(metrics['recall'])
            results['f1-score'][class_name].append(metrics['f1-score'])
    print("[INFO] Time: ", (time.time()-start)/60, "minutes.")        
    print(f"[INFO] Training completed for run {n}. \n")

final_results = {}
for metric in ['precision', 'recall', 'f1-score']:
    final_results[metric] = {}
    for class_name in results[metric].keys():
        mean = np.mean(results[metric][class_name])
        std = np.std(results[metric][class_name])
        final_results[metric][class_name] = f"{mean:.4f} +/- {std:.4f}"

accuracy_mean = np.mean(results['accuracy'])
accuracy_std = np.std(results['accuracy'])
final_results['accuracy'] = f"{accuracy_mean:.4f} +/- {accuracy_std:.4f}"

print("[INFO] Classification and vote...")
with open(os.path.join(params.path_output, 'results.json'), 'w') as f:
    json.dump(final_results, f, indent=4)      

predictions = test.predict_with_models(test_dataset, models, device, params.use_multisensors)
test.vote_and_save(predictions, test_data, inverse_species_mapping, os.path.join(params.path_output, "output.gpkg"))

print("[INFO] Classification done.")
print("[INFO] Time: ", (time.time()-start)/60, "minutes.")


# # Évaluer le modèle avec les deux jeux de données normalisés
# results_strasbourg = model.evaluate(nancy_data_normalized_strasbourg, nancy_labels)
# results_nancy = model.evaluate(nancy_data_normalized_nancy, nancy_labels)

# print(f"Performance avec normalisation Strasbourg: Loss: {results_strasbourg[0]}, Accuracy: {results_strasbourg[1]}")
# print(f"Performance avec normalisation Nancy: Loss: {results_nancy[0]}, Accuracy: {results_nancy[1]}")

#Stop carbon tracker
emissions: float = tracker.stop()
print(f'Carbon emission: {float(emissions)} kWh of electricity')
