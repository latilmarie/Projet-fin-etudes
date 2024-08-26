import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib

save = True

# _______________________________________________________________________________
## Comparison F1-scores InceptionTime and Hybrid

# F1-score for each species for InceptionTime
f1_scores_inception ={
    "Acer pseudoplatanus": 0.5821,
    "Alnus glutinosa": 0.6078,
    "Pyrus calleryana": 0.7071,
    "Prunus avium": 0.5589,
    "Tilia x euchlora": 0.7443,
    "Acer campestre": 0.5317,
    "Fraxinus excelsior": 0.5835,
    "Pinus nigra": 0.6500,
    "Platanus x acerifolia": 0.9270,
    "Carpinus betulus": 0.5722,
    "Styphnolobium japonicum": 0.8201,
    "Tilia cordata": 0.5853,
    "Populus nigra": 0.7940,
    "Robinia pseudoacacia": 0.6412,
    "Alnus x spaethii": 0.6757,
    "Acer platanoides": 0.4714,
    "Quercus robur": 0.4895,
    "Betula pendula": 0.6215,
    "Aesculus hippocastanum": 0.8135,
    "Taxus baccata": 0.6249
}

# F1-score for each species for Hybrid
f1_scores_hybrid = {
    "Acer pseudoplatanus": 0.5961,
    "Alnus glutinosa": 0.6050,
    "Pyrus calleryana": 0.7313,
    "Prunus avium": 0.5689,
    "Tilia x euchlora": 0.7588,
    "Acer campestre": 0.5725,
    "Fraxinus excelsior": 0.5885,
    "Pinus nigra": 0.6308,
    "Platanus x acerifolia": 0.9318,
    "Carpinus betulus": 0.6055,
    "Styphnolobium japonicum": 0.8309,
    "Tilia cordata": 0.6252,
    "Populus nigra": 0.7956,
    "Robinia pseudoacacia": 0.6564,
    "Alnus x spaethii": 0.6866,
    "Acer platanoides": 0.4944,
    "Quercus robur": 0.4998,
    "Betula pendula": 0.6427,
    "Aesculus hippocastanum": 0.8233,
    "Taxus baccata": 0.6386
}

f1_scores_transf = {
    "Acer pseudoplatanus": 0.5402,
    "Alnus glutinosa": 0.5812,
    "Pyrus calleryana": 0.6329,
    "Prunus avium": 0.5234,
    "Tilia x euchlora": 0.6887,
    "Acer campestre": 0.5162,
    "Fraxinus excelsior": 0.5446,
    "Pinus nigra": 0.6059,
    "Platanus x acerifolia": 0.8997,
    "Carpinus betulus": 0.5182,
    "Styphnolobium japonicum": 0.7987,
    "Tilia cordata": 0.5482,
    "Populus nigra": 0.7546,
    "Robinia pseudoacacia": 0.5869,
    "Alnus x spaethii": 0.6325,
    "Acer platanoides": 0.4432,
    "Quercus robur": 0.4229,
    "Betula pendula": 0.5679,
    "Aesculus hippocastanum": 0.7823,
    "Taxus baccata": 0.6012
}

# Tree species distribution
distribution = {
    "Acer pseudoplatanus": 619,
    "Alnus glutinosa": 162,
    "Pyrus calleryana": 195,
    "Prunus avium": 299,
    "Tilia x euchlora": 600,
    "Acer campestre": 295,
    "Fraxinus excelsior": 420,
    "Pinus nigra": 145,
    "Platanus x acerifolia": 1126,
    "Carpinus betulus": 297,
    "Styphnolobium japonicum": 158,
    "Tilia cordata": 481,
    "Populus nigra": 179,
    "Robinia pseudoacacia": 258,
    "Alnus x spaethii": 154,
    "Acer platanoides": 477,
    "Quercus robur": 150,
    "Betula pendula": 260,
    "Aesculus hippocastanum": 343,
    "Taxus baccata": 145
}

# Extract values
species = list(f1_scores_inception.keys())
f1_inception_values = np.array([f1_scores_inception[sp] for sp in species])
f1_hybrid_values = np.array([f1_scores_hybrid[sp] for sp in species])
f1_transf_values = np.array([f1_scores_transf[sp] for sp in species])
sizes = np.array([distribution[sp] for sp in species])

# Comparison graph for the 2 models
plt.figure(figsize=(12, 8))
cmap = cm.get_cmap('tab20', len(species))
colors = cmap.colors
sc = plt.scatter(f1_inception_values, f1_hybrid_values, s=sizes, c=colors, alpha=0.8, edgecolors="w", linewidth=0.5)
plt.xlabel("F1-score InceptionTime", fontsize=15)
# plt.xlabel("F1-score Transformer", fontsize=15)
# sc = plt.scatter(f1_transf_values, f1_hybrid_values, s=sizes, c=colors, alpha=0.8, edgecolors="w", linewidth=0.5)
plt.ylabel("F1-score Hybrid", fontsize=15)
plt.xlim(0.4, 1)
plt.ylim(0.4, 1)
plt.tick_params(axis = 'both', labelsize = 12)
plt.plot([0, 1], [0, 1], ls="-", color="gray")
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, alpha=0.8) for i in range(len(species))]
legend1 = plt.legend(handles, species, bbox_to_anchor=(1, 1), loc='upper left', fontsize=13, title="Tree species", title_fontsize='15', handletextpad=3.5)
size_scale = [100, 500, 1000]
size_labels = [f'{int(size)} trees' for size in size_scale]
for i in range(len(size_scale)):
    plt.scatter([], [], s=size_scale[i], color='gray', alpha=0.5, label=size_labels[i])
plt.legend(scatterpoints=1, frameon=False, labelspacing=2, title='Tree distribution', loc='upper left', fontsize=13, bbox_to_anchor=(0.02, 0.98), title_fontsize='15', handletextpad=3.5)
plt.gca().add_artist(legend1)
plt.grid(False)
plt.tight_layout(rect=[0, 0, 0.7, 1])
plt.show()
# if save:
#     plt.savefig(f"./f1_scores_comparison4.png")

# _______________________________________________________________________________
## Dates for S2 and Planet timeline

# Dates for S2 and Planet
list_dates_S2 = ['20220213', '20220223', '20220228', '20220305', '20220310', '20220320', '20220325', '20220504', '20220509', '20220514', '20220613', '20220618', '20220623', '20220703', '20220708', '20220713', '20220728', '20220807', '20220812', '20220822', '20220901', '20220921']
list_dates_Planet = ['20220119', '20220307', '20220308', '20220311', '20220321', '20220322', '20220323', '20220328', '20220417', '20220418', '20220419', '20220422', '20220428', '20220515', '20220518', '20220530', '20220610', '20220615', '20220617', '20220618', '20220621', '20220705', '20220713', '20220715', '20220717', '20220718', '20220803', '20220804', '20220806', '20220807', '20220808', '20220809', '20220810', '20220811', '20220812', '20220813', '20220815', '20220821', '20220822', '20220823', '20220904', '20220905', '20220921', '20220922', '20220923', '20221005', '20221009', '20221010', '20221017', '20221022', '20221027', '20221123', '20221231']
dates_S2 = pd.to_datetime(list_dates_S2, format='%Y%m%d')
dates_Planet = pd.to_datetime(list_dates_Planet, format='%Y%m%d')
dates_combined = sorted(dates_S2.union(dates_Planet))

# Timeline combined dates
fig = plt.figure(figsize=(12, 2))
plt.plot(dates_Planet, [1] * len(dates_Planet), '.', markersize=15, color='orange', label='Planet')
plt.plot(dates_S2, [1] * len(dates_S2), '.', markersize=15, label='S2')
plt.title('Combined dates S2 + Planet', fontsize=15)
plt.gca().get_yaxis().set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.axhline(y=1, color='gray', linestyle='--', linewidth=0.5)
xticks = pd.date_range(start=dates_combined[0], end=dates_combined[-1], periods=5)
plt.xticks(xticks, fontsize=15)
plt.legend(bbox_to_anchor=(0.9, 1.3), loc='upper left', fontsize=15)
plt.tight_layout()
plt.show()

# if save:
#     plt.savefig(f"./dates2.png")


# _______________________________________________________________________________
## Bar plot accuracy InceptionTime, Hybrid and LITE models for S2, Planet and combined

def config_plot(ax) :
    # Hide the all but the bottom spines (axis lines)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # ax.spines["bottom"].set_bounds(min(x), max(x))
    
s2_bars = [0.6292, 0.6305, 0.4574, 0.6187]
ps_bars = [0.6221, 0.6242, 0.4875, 0.6163]
combined_bars = [0.6768, 0.6913, 0.5612, 0.6373]

fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 0.1
indices = np.arange(4)/2
space = 0.02

s2_color = 'darkred'
ps_color = 'tab:blue'
combined_color = 'darkgreen'

# Positions for each bar
xl = indices - bar_width - space
xc = indices
xr = indices + bar_width + space

p2 = ax.bar(xl, s2_bars, bar_width, label='S2', color=s2_color, alpha=0.6, edgecolor='black')
p3 = ax.bar(xc, ps_bars, bar_width, label='Planet', color=ps_color, alpha=0.6, edgecolor='black')
p4 = ax.bar(xr, combined_bars, bar_width, label='S2+Planet', color=combined_color, alpha=0.6, edgecolor='black')

ax.set_ylabel('Accuracy', fontsize = 15)
ax.set_xticks(indices)
ax.tick_params(axis='both', labelsize=12)
ax.set_xticklabels(['InceptionTime', 'Hybrid', 'LITE', 'Transformer'], fontsize=15)
ax.set_ylim(0,1)

ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
legend = ['S2', 'Planet', 'S2+Planet']
ax.legend(legend, loc='upper right', bbox_to_anchor=(0.96, 0.96), fontsize=15)
config_plot(ax)
plt.tight_layout()
plt.show()
# if save:
#     plt.savefig(f"./accuracy_plot4.png")
