#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from process_data import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data(log_data=False)
    
    # Initialize RandomForestRegressor with specified parameters
    rf = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_split=15,
                         min_samples_leaf=6, max_samples=0.75, random_state=42, n_jobs=9)

    #  X_train is float64
    X_train = X_train.astype(np.float64)

    # Replace infs 
    X_train.replace([np.inf, -np.inf], 0, inplace=True)

    # Cap very large/small values to avoid float32 overflow
    X_train = X_train.clip(-1e15, 1e15)

    # Train the model on the entire training dataset
    rf.fit(X_train, y_train)

    with open('rf_trained.pkl', 'wb') as rf_pkl:
        pickle.dump(rf, rf_pkl)
        
