#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor
from tqdm import tqdm

# Function to get feature names
def get_feature_names():
    return [
        'SubhaloGasMassInMaxRad',
        'SFG',
        'Group_M_Crit200',
        'SubhaloVmaxRad',
        'SubhaloVelDisp',
    ]

# Function to define feature labels for plots
def get_feature_units():
    return {
        'SubhaloGasMassInMaxRad': r'$\log_{10}M_{\mathrm{gas, MaxRad}}\ [\mathrm{M_{\odot}}]$',
        'SFG': r'$\log_{10}M_{\mathrm{SFG}}\ [\mathrm{M_{\odot}}]$',
        'Group_M_Crit200': r'$\log_{10}M_{200}\ [\mathrm{M_{\odot}}]$',
        'SubhaloVmaxRad': r'$\log_{10}R_{\mathrm{V_{Max}}}\ [\mathrm{kpc}]$',
        'SubhaloVelDisp': r'$\log_{10}\sigma\ [\mathrm{km/s}]$'
    }

# Add units to feature names
def add_units_to_feature_names(feature_names, feature_units):
    return [feature_units[name] if name in feature_units else name for name in feature_names]

# Function to load and preprocess data
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

    # Escape velocities
    df100['EscapeVelocity1'] = np.sqrt(2 * G * M1 / r1)
    df100['EscapeVelocity2'] = np.sqrt(2 * G * M2 / r2)
    df100['EscapeVelocity3'] = np.sqrt(2 * G * M3 / r3)
    df100['EscapeVelocity4'] = np.sqrt(2 * G * M4 / r4)
    df100['EscapeVelocity5'] = np.sqrt(2 * G * M1 / r5)

    # Gravitational potentials
    df100['GravitationalPotential_M1'] = G * M1 / r1
    df100['GravitationalPotential_M2'] = G * M2 / r2
    df100['GravitationalPotential_M3'] = G * M3 / r3
    df100['GravitationalPotential_M4'] = G * M4 / r4
    df100['GravitationalPotential_r5'] = G * M1 / r5

    df100.to_pickle("full_illustris_df100_with_escape_velocity.pkl")

    f_bar = (df100['GroupMass'] - df100['GroupDMMass']) / (0.15733 * df100['GroupMass'])
    f_bar_scaled = f_bar * 0.15733

    drop_features = [
        'QuasarEnergy', 'SubhaloMass', 'GroupMass', 'GroupDMMass', 'GroupGasMass', 'GroupStarMass',
        'SubhaloGasMass', 'SubhaloDMMass', 'SubhaloBHMass', 'SubhaloStarMass', 'SubhaloMassInRad', 
        'SubhaloMassInHalfRad', 'SubhaloMassInMaxRad', 'SubhaloStarMassInMaxRad', 'SubhaloDMMassInMaxRad', 
        'SubhaloDMMassInHalfRad', 'SubhaloDMMassInRad', 'SubhaloHalfmassRad', 'Group_R_Crit200',
        'EscapeVelocity4', 'SubhaloGasHalfmassRad', 'GravitationalPotential_M4', 'GroupSFR'
    ]
    features = df100.drop(columns=drop_features)

    X_train, X_test, y_train, y_test = train_test_split(features, f_bar_scaled, test_size=0.2, random_state=42)

    if log_data:
        for dataset in [X_train, X_test]:
            for feature in dataset.columns:
                if dataset[feature].min() == 0:
                    dataset.loc[dataset[feature] == 0, feature] = dataset[dataset[feature] != 0].min() / 1e4
                dataset[feature] = dataset[feature].apply(np.log10)

    return X_train, X_test, y_train, y_test, features, f_bar_scaled

# Log-transform data
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    for feature in df.columns:
        if df[feature].min() == 0:
            df.loc[df[feature] == 0, feature] = df[df[feature] != 0].min() * 10**-4
        df[feature] = df[feature].apply(np.log10)
    return df

# Main
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, f_bar_scaled = get_train_test_data()
    feature_names = get_feature_names()
    feature_units = get_feature_units()

    random_states = np.random.randint(0, 100, 10)
    models = []

    for state in tqdm(random_states):
        X_train, X_test, y_train, y_test = train_test_split(features, f_bar_scaled, test_size=0.2, random_state=state)

        X_train_selected = transform_data(X_train[feature_names])
        X_test_selected = transform_data(X_test[feature_names])

        ebm = ExplainableBoostingRegressor(
            early_stopping_rounds=50, inner_bags=8, interactions=20, learning_rate=0.6,
            max_bins=256, max_interaction_bins=32, max_leaves=2, max_rounds=25000,
            min_samples_leaf=20, outer_bags=14, smoothing_rounds=2000,
            feature_types=['quantile'] * len(feature_names), random_state=state
        )

        ebm.fit(X_train_selected, y_train)
        models.append(ebm)

    selected_features_with_units = add_units_to_feature_names(feature_names, feature_units)

    # Subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(len(models[0].feature_names_in_)):
        ax = axes[i]

        all_x = [model.bins_[i][0] for model in models]
        all_y = [model.term_scores_[i][1:-2] for model in models]

        x_vals = all_x[0]
        y_vals_matrix = np.array(all_y)

        y_mean = np.mean(y_vals_matrix, axis=0)
        y_min = np.min(y_vals_matrix, axis=0)
        y_max = np.max(y_vals_matrix, axis=0)

        # Shaded region between min and max
        ax.fill_between(x_vals, y_min, y_max, color='purple', alpha=0.3, label='Min–Max Range')
        ax.plot(x_vals, y_mean, color='black', linewidth=2, label='Mean')

        ax.axhline(0, color='blue', linestyle='--', linewidth=0.8)

        ylim = ax.get_ylim()
        ax.axhspan(0, ylim[1], facecolor='lightblue', alpha=0.3, zorder=-1)
        ax.axhspan(ylim[0], 0, facecolor='lightcoral', alpha=0.3, zorder=-1)

        ax.set_xlabel(selected_features_with_units[i], fontweight='bold')
        ax.set_ylabel(r'$f_{\mathrm{Bar}}$', fontweight='bold')

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        if i == 0:
            ax.legend()

    if len(models[0].feature_names_in_) < len(axes):
        fig.delaxes(axes[-1])

    fig.tight_layout()
    plt.savefig('ebm_multiple_sum.pdf')
    plt.show()
