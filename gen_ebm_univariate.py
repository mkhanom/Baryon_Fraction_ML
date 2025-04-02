#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess_data import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from interpret.glassbox import ExplainableBoostingRegressor

if __name__ == "__main__":
    # Load train and test data
    X_train, X_test, y_train, y_test = get_train_test_data()
    feature_names = get_feature_names()

    # Load the trained EBM model
    with open("ebm_trained.pkl", 'rb') as ebm_pkl:
        ebm = pickle.load(ebm_pkl)

# LaTeX-formatted feature names for the x-axis
variables = [
    r'$\log_{10}\sigma\ [\mathrm{km/s}]$',  
    r'$\log_{10}R_{\mathrm{V_{Max}}}\ [\mathrm{kpc}]$', 
    r'$\log_{10}M_{200}\ [\mathrm{M_{\odot}}]$', 
    r'$\log_{10}M_{\mathrm{SFG}}\ [\mathrm{M_{\odot}}]$',
    r'$\log_{10}M_{\mathrm{gas, MaxRad}}\ [\mathrm{M_{\odot}}]$',
]

# Reverse the variables list
variables_reversed = variables[::-1]

# Create subplots: 2 rows and 3 columns
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()

# Reverse the order of features explicitly
for i, (variable, feature_index) in enumerate(zip(variables_reversed, range(len(variables) - 1, -1, -1))):
    # Access bins and term scores in the reversed order
    bins = ebm.bins_[feature_index][0]
    scores = ebm.term_scores_[feature_index][1:-2]

    # Plot the term scores for each feature
    axes[i].plot(bins, scores, color='black')

    # Add a horizontal dashed line at zero
    axes[i].axhline(0, color='blue', linestyle='--', linewidth=0.5)

    # Shade regions above and below zero
    ylim = axes[i].get_ylim()  # Get y-axis limits
    axes[i].axhspan(0, ylim[1], facecolor='lightblue', alpha=0.5, zorder=-1)  # Shade above zero
    axes[i].axhspan(ylim[0], 0, facecolor='lightcoral', alpha=0.5, zorder=-1)  # Shade below zero

    # Set Y-axis label with bold font
    axes[i].set_ylabel(r'$f_{Bar}$', fontweight='bold')  # Y-axis label

    # Set X-axis label with LaTeX-formatted variable names and bold font
    axes[i].set_xlabel(variable, fontweight='bold')  # X-axis label

    # Make plot borders bold by adjusting the line width
    for spine in axes[i].spines.values():
        spine.set_linewidth(2)  # Bold borders with line width set to 2

# Remove the last empty subplot
fig.delaxes(axes[-1])

# Adjust layout for better spacing
fig.tight_layout()

# Save the figure as a PDF file
plt.savefig('ebm_Univariate_shaded1.pdf')

# Display the plot
plt.show()
