#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        'SubhaloGasMassInMaxRad': r'$\log_{10}M_{\mathrm{gas, MaxRad}}\ [\mathrm{M_{\odot}}]$'
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data()
    feature_names = get_feature_names()

    with open("ebm_trained.pkl", 'rb') as ebm_pkl:
        ebm = pickle.load(ebm_pkl)

    df100 = pd.read_pickle("full_illustris_df100.pkl")

    # Specified features
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

    for state in tqdm(random_states):
        X_train, X_test, y_train, y_test = train_test_split(features, f_bar_scaled, test_size=0.2, random_state=state)

        ebm = ExplainableBoostingRegressor(
            early_stopping_rounds=50, inner_bags=8, interactions=10, learning_rate=0.6, 
            max_bins=256, max_interaction_bins=32, max_leaves=2, max_rounds=25000, 
            min_samples_leaf=20, outer_bags=14, smoothing_rounds=2000, 
            feature_types=['quantile'] * len(selected_features), random_state=state
        )

        X_train = transform_data(X_train)
        X_test = transform_data(X_test)

        ebm.fit(X_train, y_train)
        models.append(ebm)

    feature_units = get_feature_units()

    model_it_scores = []
    for model in models:
        it_scores = np.vstack([score[np.newaxis, ...] for score in model.term_scores_[5:]])[np.newaxis, ...]
        model_it_scores.append(it_scores)
    model_it_scores = np.vstack(model_it_scores)
    it_std = np.std(model_it_scores, axis=0)
    vmin, vmax = it_std.min(), it_std.max()

    # Extract interaction features
    it_feat = [f for f in models[0].term_features_ if len(f) > 1]

    # Specify best plots (1 and 8)
    indices_to_plot = [1, 8]

    # Create subplots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    for i, subplot_index in enumerate(indices_to_plot):
        idx, idy = it_feat[subplot_index]
        im = ax[i].imshow(
            it_std[subplot_index], cmap='bwr', aspect='auto', vmax=vmax, vmin=vmin,
            extent=[
                models[0].bins_[idx][0][0], models[0].bins_[idx][0][-1], 
                models[0].bins_[idy][0][0], models[0].bins_[idy][0][-1]
            ]
        )

        colnameX = models[0].feature_names_in_[idx]
        colnameY = models[0].feature_names_in_[idy]

        # Add labels with feature units
        ax[i].set_xlabel(feature_units.get(colnameX, colnameX))  # Use units if available
        ax[i].set_ylabel(feature_units.get(colnameY, colnameY))  # Use units if available

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])  # Colorbar position
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_title('std')

    # Save and display the plot
    plt.savefig('best_bistd.pdf', bbox_inches='tight')  # Use bbox_inches to avoid tight_layout issues
    plt.show()
