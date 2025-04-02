
#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess_data import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from interpret.glassbox import ExplainableBoostingRegressor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data()
    feature_names = get_feature_names()

    # Corrected best parameters found by GridSearchCV
    best_params = {'early_stopping_rounds': 50, 'inner_bags': 8,
                   'interactions': 20, 'learning_rate': 0.6, 'max_bins': 256,
                   'max_interaction_bins': 32, 'max_leaves': 2, 'max_rounds':
                   25000, 'min_samples_leaf': 20, 'outer_bags': 14,
                   'smoothing_rounds': 2000}

    # Initialization of the model with the best parameters
    ebm = ExplainableBoostingRegressor(**best_params)

    # Fitting the model on the training dataset
    ebm.fit(X_train[feature_names], y_train)

    with open('ebm_trained.pkl', 'wb') as ebm_pkl:
        pickle.dump(ebm, ebm_pkl)

# Plot the results
    plt.plot(y_test, ebm.predict(X_test), 'k,')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', 
             label=f'$R^2= {r2_score(y_test, ebm.predict(X_test)):.3f}$')
    plt.legend()
    plt.xlabel('True Baryon Fraction')
    plt.ylabel('Predicted Baryon Fraction')
    plt.savefig('ebm_accuracy1.pdf')

plt.savefig('ebm_accuracy.pdf')
        
