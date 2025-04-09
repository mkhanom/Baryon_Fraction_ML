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
    df100 = pd.read_pickle("/home/mkhanom/Final_Model_T/full_illustris_df100.pkl")

    G = 4.302e-6  # kpc (km/s)^2 M_sun^-1

    # Avoid divide-by-zero issues
    df100['SubhaloStarHalfmassRad'] = df100['SubhaloStarHalfmassRad'].replace(0, 1e-6)
    df100['SubhaloVmaxRad'] = df100['SubhaloVmaxRad'].replace(0, 1e-6)
    df100['Group_R_Crit200'] = df100['Group_R_Crit200'].replace(0, 1e-6)
    df100['SubhaloGasHalfmassRad'] = df100['SubhaloGasHalfmassRad'].replace(0, 1e-6)

    # Masses and radii
    M1 = df100['SubhaloMassInHalfRad']
    M2 = df100['SubhaloMassInMaxRad']
    M3 = df100['SubhaloMassInRad']
    M4 = df100['Group_M_Crit200']
    r1 = df100['SubhaloStarHalfmassRad']
    r2 = df100['SubhaloVmaxRad']
    r3 = 2 * df100['SubhaloStarHalfmassRad']
    r4 = df100['Group_R_Crit200']
    r5 = df100['SubhaloGasHalfmassRad']

    # Escape velocities and gravitational potentials
    df100['EscapeVelocity1'] = np.sqrt(2 * G * M1 / r1)
    df100['EscapeVelocity2'] = np.sqrt(2 * G * M2 / r2)
    df100['EscapeVelocity3'] = np.sqrt(2 * G * M3 / r3)
    df100['EscapeVelocity4'] = np.sqrt(2 * G * M4 / r4)
    df100['EscapeVelocity5'] = np.sqrt(2 * G * M1 / r5)

    df100['GravitationalPotential_M1'] = G * M1 / r1
    df100['GravitationalPotential_M2'] = G * M2 / r2
    df100['GravitationalPotential_M3'] = G * M3 / r3
    df100['GravitationalPotential_M4'] = G * M4 / r4
    df100['GravitationalPotential_r5'] = G * M1 / r5

    # Optional: save for inspection
    df100.to_pickle("full_illustris_df100_with_escape_velocity.pkl")

    # Target variable: scaled baryon fraction
    f_bar = (df100['GroupMass'] - df100['GroupDMMass']) / (0.15733 * df100['GroupMass'])
    f_bar_scaled = f_bar * 0.15733

    # Drop columns
    features = df100.drop(
        [
            'QuasarEnergy', 'SubhaloMass', 'GroupMass', 'GroupDMMass', 'GroupGasMass', 'GroupStarMass',
            'SubhaloGasMass', 'SubhaloDMMass', 'SubhaloBHMass', 'SubhaloStarMass', 'SubhaloMassInRad',
            'SubhaloMassInHalfRad', 'SubhaloMassInMaxRad', 'SubhaloStarMassInMaxRad', 'SubhaloDMMassInMaxRad',
            'SubhaloDMMassInHalfRad', 'SubhaloDMMassInRad', 'SubhaloHalfmassRad', 'Group_R_Crit200',
            'EscapeVelocity4', 'SubhaloGasHalfmassRad', 'GravitationalPotential_M4', 'GroupSFR'
        ],
        axis='columns'
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, f_bar_scaled, random_state=42)

    # Clean values (inf, NaN, too large)
    for df in [X_train, X_test]:
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        df[:] = df.where(df <= 1e15, 1e15)
        df[:] = df.where(df >= -1e15, -1e15)

    # log transform
    if log_data:
        for df in [X_train, X_test]:
            for col in df.columns:
                if df[col].min() <= 0:
                    min_nonzero = df[col][df[col] > 0].min()
                    df[col] = df[col].replace(0, min_nonzero / 1e4)
                df[col] = np.log10(df[col])

    return X_train, X_test, y_train, y_test


def transform_data(df: pd.DataFrame, exclude: list = []) -> pd.DataFrame:
   
    for col in df.columns:
        if col not in exclude:
            if df[col].min() <= 0:
                min_nonzero = df[col][df[col] > 0].min()
                df[col] = df[col].replace(0, min_nonzero / 1e4)
            df[col] = np.log10(df[col])
    return df


    
   

    
