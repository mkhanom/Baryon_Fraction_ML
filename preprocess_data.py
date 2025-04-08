#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_feature_names():
    return [
        'SubhaloVelDisp',
        'SubhaloVmaxRad',
        'Group_M_Crit200',
        'SFG',
        'SubhaloGasMassInMaxRad'
    ]

def get_train_test_data(log_data=True):
    # Load the data
    df100 = pd.read_pickle("full_illustris_df100.pkl")

    # Gravitational constant with units
    G = 4.302e-6  # kpc (km/s)^2 M_sun^-1

    # Masses to calculate gravitational potentials and escape velocities
    M1 = df100['SubhaloMassInHalfRad']
    M2 = df100['SubhaloMassInMaxRad']
    M3 = df100['SubhaloMassInRad']
    M4 = df100['Group_M_Crit200']

    # Radius
    r1 = df100['SubhaloStarHalfmassRad']
    r2 = df100['SubhaloVmaxRad']
    r3 = 2 * df100['SubhaloStarHalfmassRad']
    r4 = df100['Group_R_Crit200']
    r5 = df100['SubhaloGasHalfmassRad']

    # Calculation of the escape velocities
    v_esc1 = np.sqrt(2 * G * M1 / r1)
    v_esc2 = np.sqrt(2 * G * M2 / r2)
    v_esc3 = np.sqrt(2 * G * M3 / r3)
    v_esc4 = np.sqrt(2 * G * M4 / r4)
    v_esc5 = np.sqrt(2 * G * M1 / r5)

    # Calculation of the gravitational potentials
    Phi1 = G * M1 / r1
    Phi2 = G * M2 / r2
    Phi3 = G * M3 / r3
    Phi4 = G * M4 / r4
    Phi5 = G * M1 / r5

    # Adding to the DataFrame
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

    # Saving the updated DataFrame
    df100.to_pickle("full_illustris_df100_with_escape_velocity.pkl")

    # Calculate scaled baryon fraction
    f_bar = (df100['GroupMass'] - df100['GroupDMMass']) / (0.15733 * df100['GroupMass'])
    f_bar_scaled = f_bar * 0.15733

    # Select features by dropping specific columns that are not relevant or redundant
    features = df100.drop(
        [
            'QuasarEnergy', 'SubhaloMass', 'GroupMass', 'GroupDMMass', 'GroupGasMass', 'GroupStarMass',
            'SubhaloGasMass', 'SubhaloDMMass', 'SubhaloBHMass', 'SubhaloStarMass', 'SubhaloMassInRad',
            'SubhaloMassInHalfRad', 'SubhaloMassInMaxRad', 'SubhaloStarMassInMaxRad', 'SubhaloDMMassInMaxRad',
            'SubhaloDMMassInHalfRad', 'SubhaloDMMassInRad', 'SubhaloHalfmassRad', 'Group_R_Crit200',
            'EscapeVelocity4', 'SubhaloGasHalfmassRad', 'GravitationalPotential_M4', 'GroupSFR'
        ], axis='columns')

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, f_bar_scaled, random_state=42)

    exclude = []  # List of features to exclude from log transformation

    # Apply log transformation to the data if log_data is True
    if log_data:
        for feature in X_train.columns:
            if X_train[feature].min() == 0 and feature not in exclude:
                X_train.loc[X_train[feature] == 0, feature] = X_train[X_train[feature] != 0].min() / 1e4
            X_train[feature] = X_train[feature].apply(np.log10)

        for feature in X_test.columns:
            if X_test[feature].min() == 0 and feature not in exclude:
                X_test.loc[X_test[feature] == 0, feature] = X_test[X_test[feature] != 0].min() / 1e4
            X_test[feature] = X_test[feature].apply(np.log10)

    return X_train, X_test, y_train, y_test

def transform_data(df: pd.DataFrame, exclude: list = []) -> pd.DataFrame:
    for feature in df.columns:
        if feature not in exclude:
            if df[feature].min() == 0:
                df.loc[df[feature] == 0, feature] = df[df[feature] != 0].min() / 1e4
            df[feature] = df[feature].apply(np.log10)
    return df
