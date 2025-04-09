
#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from process_data import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


if __name__ == "__main__":

 
    # Get the training and testing data
    X_train, X_test, y_train, y_test = get_train_test_data(log_data=False)
 
    # Ensure float64 and clean values in X_train
    X_train = X_train.astype(np.float64)
    X_train.replace([np.inf, -np.inf], 0, inplace=True)
    X_train.fillna(0, inplace=True)
    X_train = X_train.clip(-1e15, 1e15)

    # Initialize RandomForestRegressor with specified parameters
    rf = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_split=15,
                               min_samples_leaf=6, max_samples=0.75,random_state=42, n_jobs=9)
     
  
    # Train the model on the entire training dataset
    rf.fit(X_train, y_train)
    
    # Save the trained model to a file
    with open('rf_trained.pkl', 'wb') as rf_pkl:
        pickle.dump(rf, rf_pkl)
        
    # Clean X_test the same way
    X_test = X_test.astype(np.float64)
    X_test.replace([np.inf, -np.inf], 0, inplace=True)
    X_test.fillna(0, inplace=True)
    X_test = X_test.clip(-1e15, 1e15)
    
   # Predict and plot
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    plt.plot(y_test, y_pred, 'k,')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', label=f'$R^2 = {r2:.3f}$')
    plt.legend()
    plt.xlabel('True Baryon Fraction')
    plt.ylabel('Predicted Baryon Fraction')
    plt.savefig('rf_accuracy.pdf')

