#!/usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from preprocess_data import get_train_test_data, get_feature_names, transform_data
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor
from tqdm import tqdm


def get_feature_units():
    return {
        'SubhaloVelDisp': r'$\log_{10}\sigma\ [\mathrm{km/s}]$',
        'SubhaloVmaxRad': r'$\log_{10}R_{\mathrm{V_{Max}}}\ [\mathrm{kpc}]$',
        'Group_M_Crit200': r'$\log_{10}M_{200}\ [\mathrm{M_{\odot}}]$',
        'SFG': r'$\log_{10}M_{\mathrm{SFG}}\ [\mathrm{M_{\odot}}]$',
        'SubhaloGasMassInMaxRad': r'$\log_{10}M_{\mathrm{gas,MaxRad}}\ [\mathrm{M_{\odot}}]$'
    }


def add_units_to_feature_names(feature_names, feature_units):
    return [f"{name} ({feature_units.get(name, name)})" for name in feature_names]


if __name__ == "__main__":
    # Load data and initialize variables
    df100 = pd.read_pickle("/home/mkhanom/Final_Model_T/full_illustris_df100.pkl")

    # Specify selected features and process target variable
    selected_features = [
        "SubhaloVelDisp",
        "SubhaloVmaxRad",
        "Group_M_Crit200",
        "SFG",
        "SubhaloGasMassInMaxRad",
    ]
    features = df100[selected_features].copy()
    f_bar = (df100['GroupMass'] - df100['GroupDMMass']) / (0.15733 * df100['GroupMass'])
    f_bar_scaled = f_bar * 0.15733

    random_states = np.random.randint(0, 100, 10)
    models = []

    # Train EBM models
    for state in tqdm(random_states):
        X_train, X_test, y_train, y_test = train_test_split(features, f_bar_scaled, test_size=0.2, random_state=state)

        ebm = ExplainableBoostingRegressor(
            early_stopping_rounds=50,
            inner_bags=8,
            interactions=10,
            learning_rate=0.6,
            max_bins=256,
            max_interaction_bins=32,
            max_leaves=2,
            max_rounds=25000,
            min_samples_leaf=20,
            outer_bags=14,
            smoothing_rounds=2000,
            feature_types=['quantile'] * len(selected_features),
            random_state=state
        )

        X_train = transform_data(X_train)
        X_test = transform_data(X_test)
        ebm.fit(X_train, y_train)
        models.append(ebm)

    # Add units to feature names
    feature_units = get_feature_units()
    selected_features_with_units = add_units_to_feature_names(selected_features, feature_units)

    # Aggregate interaction scores
    model_it_scores = []
    for model in models:
        it_scores = np.vstack([score[np.newaxis, ...] for score in model.term_scores_[5:]])[np.newaxis, ...]
        model_it_scores.append(it_scores)
    model_it_scores = np.vstack(model_it_scores)

    # Calculate standard deviation and normalize
    std_dev = np.std(model_it_scores, axis=0)
    std_dev[std_dev == 0] = np.nan
    it_std = np.mean(model_it_scores, axis=0) / std_dev
    it_std = np.nan_to_num(it_std, nan=0.0)

    # Define the colormap
    min_val, max_val = -20, 30
    zero_norm = (0 - min_val) / (max_val - min_val)
    cmap = LinearSegmentedColormap.from_list('custom_blue_red', [(0, 'blue'), (zero_norm, 'white'), (1, 'red')])

    # Indices for the subplots to display (1 and 8, zero-based)
    indices_to_plot = [1, 8]
    it_feat = [f for f in models[0].term_features_ if len(f) > 1]

    # Create subplots for selected indices
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax = ax.flatten()

    for i, subplot_index in enumerate(indices_to_plot):
        idx, idy = it_feat[subplot_index]
        im = ax[i].imshow(
            it_std[subplot_index],
            cmap=cmap,
            aspect='auto',
            vmin=min_val,
            vmax=max_val,
            extent=[
                models[0].bins_[idx][0][0], models[0].bins_[idx][0][-1],
                models[0].bins_[idy][0][0], models[0].bins_[idy][0][-1]
            ]
        )
        colnameX, colnameY = models[0].feature_names_in_[idx], models[0].feature_names_in_[idy]
        colnameX_label = feature_units.get(colnameX, colnameX)
        colnameY_label = feature_units.get(colnameY, colnameY)
        ax[i].set_xlabel(colnameX_label)
        ax[i].set_ylabel(colnameY_label)

    # Adjust layout and save plot
    fig.tight_layout(pad=2.5)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_title('Z-Score')

    plt.savefig('ebm_z_best.pdf')
    plt.show()
