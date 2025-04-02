#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data for the feature importances across 10 models
data = {
    "model": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 5,
    "variable": ["SubhaloVelDisp"] * 10 + ["SubhaloVmaxRad"] * 10 + ["Group_M_Crit200"] * 10 +
                ["SFG"] * 10 + ["SubhaloGasMassInMaxRad"] * 10,
    "value": [
        0.071396, 0.070034, 0.069641, 0.069997, 0.070239, 0.068684, 0.070419, 0.069665, 0.072617, 0.069648,
        0.052155, 0.052130, 0.052190, 0.052202, 0.052200, 0.051899, 0.052454, 0.051977, 0.052320, 0.052382,
        0.017109, 0.016798, 0.016748, 0.017659, 0.016469, 0.016796, 0.017141, 0.016120, 0.017580, 0.017386,
        0.052371, 0.051350, 0.050753, 0.051556, 0.049721, 0.051299, 0.051729, 0.051789, 0.052686, 0.052521,
        0.181717, 0.180798, 0.180863, 0.180914, 0.182513, 0.180885, 0.181567, 0.181128, 0.182566, 0.182677
    ]
}

# Created a DataFrame
df = pd.DataFrame(data)

# Mapping features to their equation-format labels
feature_labels = {
    'SubhaloGasMassInMaxRad': r'$\log_{10}M_{\mathrm{gas, MaxRad}}\ [\mathrm{M_{\odot}}]$',
    'SFG': r'$\log_{10}M_{\mathrm{SFG}}\ [\mathrm{M_{\odot}}]$',
    'Group_M_Crit200': r'$\log_{10}M_{200}\ [\mathrm{M_{\odot}}]$',
    'SubhaloVmaxRad': r'$\log_{10}R_{\mathrm{V_{Max}}}\ [\mathrm{kpc}]$',
    'SubhaloVelDisp': r'$\log_{10}\sigma\ [\mathrm{km/s}]$'
}

# Replacing variable names with LaTeX labels
df['variable_latex'] = df['variable'].map(feature_labels)

# Center values and scale to reduce box height
df['value_centered'] = df.groupby('variable_latex')['value'].transform(lambda x: x - x.mean())
df['value_scaled'] = df['value_centered'] * 0.1  # can adjust this factor to control height

# Plot the scaled values
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="variable_latex", y="value_scaled")
plt.xlabel("Feature")
plt.ylabel(r"Importance $-$ $\overline{\mathrm{Importance}}$")
#plt.ylim(-0.0025, 0.0025)  # Adjust as needed based on the scaling factor
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ebm_importance_scaled.pdf')
plt.show()
