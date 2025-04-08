#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from interpret.glassbox import ExplainableBoostingRegressor
from matplotlib.colors import Normalize, LinearSegmentedColormap


def get_feature_names():
    return [
        "SubhaloGasMassInMaxRad",
        "SFG",
        "Group_M_Crit200",
        "SubhaloVmaxRad",
        "SubhaloVelDisp",
    ]


def get_feature_units():
    return {
        'SubhaloGasMassInMaxRad': r'\log_{10}M_{\mathrm{gas, MaxRad}}\ [\mathrm{M_{\odot}}]',
        'SFG': r'\log_{10}M_{\mathrm{SFG}}\ [\mathrm{M_{\odot}}]',
        'Group_M_Crit200': r'\log_{10}M_{200}\ [\mathrm{M_{\odot}}]',
        'SubhaloVmaxRad': r'\log_{10}R_{\mathrm{V_{Max}}}\ [\mathrm{kpc}]',
        'SubhaloVelDisp': r'\log_{10}\sigma\ [\mathrm{km/s}]',
    }


def get_train_test_data(log_data=True):
    df100 = pd.read_pickle("full_illustris_df100.pkl")
    G = 4.302e-6  # kpc (km/s)^2 M_sun^-1

    M1 = df100['SubhaloMassInHalfRad']
    M2 = df100['SubhaloMassInMaxRad']
    M3 = df100['SubhaloMassInRad']
    M4 = df100['Group_M_Crit200']

    r1 = df100['SubhaloStarHalfmassRad']
    r2 = df100['SubhaloVmaxRad']
    r3 = 2 * df100['SubhaloStarHalfmassRad']
    r4 = df100['Group_R_Crit200']
    r5 = df100['SubhaloGasHalfmassRad']

    v_esc1 = np.sqrt(2 * G * M1 / r1)
    v_esc2 = np.sqrt(2 * G * M2 / r2)
    v_esc3 = np.sqrt(2 * G * M3 / r3)
    v_esc4 = np.sqrt(2 * G * M4 / r4)
    v_esc5 = np.sqrt(2 * G * M1 / r5)

    Phi1 = G * M1 / r1
    Phi2 = G * M2 / r2
    Phi3 = G * M3 / r3
    Phi4 = G * M4 / r4
    Phi5 = G * M1 / r5

    df100['EscapeVelocity1'] = v_esc1
    df100['EscapeVelocity2'] = v_esc2
    df100['EscapeVelocity3'] = v_esc3
    df100['EscapeVelocity4'] = v_esc4
    df100['EscapeVelocity5'] = v_esc5
    df100['GravitationalPotential_M1'] = Phi1
    df100['GravitationalPotential_M2'] = Phi2
    df100['GravitationalPotential_M3'] = Phi3
    df100['GravitationalPotential_M4'] = Phi4
    df100['GravitationalPotential_r5'] = Phi5

    df100.to_pickle("full_illustris_df100_with_escape_velocity.pkl")

    f_bar = (df100['GroupMass'] - df100['GroupDMMass']) / (0.15733 * df100['GroupMass'])
    f_bar_scaled = f_bar * 0.15733

    features = df100.drop(
        [
            'QuasarEnergy', 'SubhaloMass', 'GroupMass', 'GroupDMMass', 'GroupGasMass', 'GroupStarMass',
            'SubhaloGasMass', 'SubhaloDMMass', 'SubhaloBHMass', 'SubhaloStarMass', 'SubhaloMassInRad',
            'SubhaloMassInHalfRad', 'SubhaloMassInMaxRad', 'SubhaloStarMassInMaxRad', 'SubhaloDMMassInMaxRad',
            'SubhaloDMMassInHalfRad', 'SubhaloDMMassInRad', 'SubhaloHalfmassRad', 'Group_R_Crit200',
            'EscapeVelocity4', 'SubhaloGasHalfmassRad', 'GravitationalPotential_M4', 'GroupSFR'
        ], axis='columns')

    X_train, X_test, y_train, y_test = train_test_split(features, f_bar_scaled, test_size=0.2, random_state=42)

    if log_data:
        exclude = []
        for feature in X_train.columns:
            if X_train[feature].min() == 0 and feature not in exclude:
                X_train.loc[X_train[feature] == 0, feature] = X_train[X_train[feature] != 0].min() / 1e4
            X_train[feature] = X_train[feature].apply(np.log10)

        for feature in X_test.columns:
            if X_test[feature].min() == 0 and feature not in exclude:
                X_test.loc[X_test[feature] == 0, feature] = X_test[X_test[feature] != 0].min() / 1e4
            X_test[feature] = X_test[feature].apply(np.log10)

    return X_train, X_test, y_train, y_test, features, f_bar_scaled


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, f_bar_scaled = get_train_test_data()
    feature_names = get_feature_names()
    feature_units = get_feature_units()

    with open("ebm_trained.pkl", 'rb') as ebm_pkl:
        ebm = pickle.load(ebm_pkl)

    min_val, max_val = -0.02, 0.015
    norm = Normalize(vmin=min_val, vmax=max_val)

    cmap = LinearSegmentedColormap.from_list(
        'custom_blue_red',
        [(0, 'blue'), (abs(min_val) / (max_val + abs(min_val)), 'white'), (1, 'red')]
    )

    specific_ids = [1, 8]
    it_feat = [f for f in ebm.term_features_ if len(f) > 1]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    it_scores = ebm.term_scores_[5:]

    for plot_idx, (ax, j) in enumerate(zip(axes, specific_ids)):
        idx, idy = it_feat[j]
        score = it_scores[j]

        # Plot heatmap
        im = ax.imshow(score, cmap=cmap, aspect='auto', norm=norm,
                       extent=[ebm.bins_[idx][0][0], ebm.bins_[idx][0][-1],
                               ebm.bins_[idy][0][0], ebm.bins_[idy][0][-1]])

        colnameX = ebm.feature_names_in_[idx]
        colnameY = ebm.feature_names_in_[idy]
        ax.set_xlabel(f'${feature_units.get(colnameX, colnameX)}$')
        ax.set_ylabel(f'${feature_units.get(colnameY, colnameY)}$')

        # Add green triangle only to second plot
        if plot_idx == 1:
            xmin = ebm.bins_[idx][0][0]
            xmax = ebm.bins_[idx][0][-1]
            ymin = ebm.bins_[idy][0][0]
            ymax = ebm.bins_[idy][0][-1]

            triangle_x = [xmin, 10.5, xmin]
            triangle_y = [ymax, ymax, 8.9]
            ax.fill(triangle_x, triangle_y, color='black', alpha=0.4, edgecolor='none')

    fig.tight_layout(pad=2.5)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_title('Baryon Fraction')

    plt.savefig('ebm_bivariate_with_triangle_matched.pdf')
    plt.show()
