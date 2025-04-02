#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess_data import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data(log_data=False)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    # Initialize RandomForestRegressor with specified parameters
    rf = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_split=15,
                         min_samples_leaf=6, max_samples=0.75, random_state=42, n_jobs=9)

    # Train the model on the entire training dataset
    rf.fit(X_train, y_train)

    with open('rf_trained.pkl', 'wb') as rf_pkl:
        pickle.dump(rf, rf_pkl)
        