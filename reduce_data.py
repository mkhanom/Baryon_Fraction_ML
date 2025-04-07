#!/usr/bin/env python
# # Importing Necessary Libraries
import numpy as np
import pandas as pd
import illustris_python as il
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import seaborn as sns
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import os
import h5py

from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

basePath = '/home/kellerbw/illustris/TNG100-1' 
little_h = 0.6774
M_unit = 1e10/little_h


# # Data Preprocessing
# Group level
# 
# Has stellar mass > 0,
# Has gas mass > 0,
# Has total group mass of at least 1000 DM particles ( >4.5×10^9𝑀⊙),
# Has total baryon fraction less than cosmological  (<0.15733),
# 
# 
# SubHalo level
# 
# SubhaloFlag == 1,
# Has stellar mass > 0,
# Has gas mass > 0

# ## Data Loading
## Fields with units of mass
groupMassFields = ['Group_M_Crit200', 'GroupMass', 'GroupBHMass', 'GroupBHMdot', 'GroupMassType']
groupFields = ['GroupFirstSub', 'Group_R_Crit200', 'GroupStarMetallicity', 'GroupGasMetallicity', 
               'GroupSFR', 'GroupGasMetalFractions', 'GroupStarMetalFractions']+groupMassFields
group_data = il.groupcat.loadHalos(basePath, 99, groupFields)
subhaloMassFields = ['SubhaloMass','SubhaloMassType', 'SubhaloBHMass', 'SubhaloMassInRadType', 'SubhaloMassInHalfRadType', 'SubhaloMassInMaxRadType']
subhaloFields = ['SubhaloFlag', 'SubhaloGasMetalFractions', 'SubhaloMassInRad','SubhaloMassInHalfRad','SubhaloMassInMaxRad','SubhaloHalfmassRad', 'SubhaloSFR','SubhaloVmax', 'SubhaloGasMetallicity', 'SubhaloStarMetallicity', 'SubhaloStarMetalFractions','SubhaloBfldDisk','SubhaloVmaxRad',
                'SubhaloStarMetallicityMaxRad','SubhaloGasMetallicityMaxRad','SubhaloGasMetallicityHalfRad','SubhaloStarMetallicityHalfRad','SubhaloVelDisp','SubhaloBfldHalo','SubhaloGasMetallicitySfr','SubhaloHalfmassRadType'] + subhaloMassFields
subhaloFields.append('SubhaloBfldHalo')
subhalo_data = il.groupcat.loadSubhalos(basePath, 99, subhaloFields)
df_stellar_merger_history = pd.read_csv('stellar_merge_history.csv', index_col=0)

for col in df_stellar_merger_history.columns:
    subhalo_data[col] = df_stellar_merger_history[col].values
    subhaloFields.append(col)

group_conditions = (group_data['GroupMass']*M_unit > 7.5e9, group_data['GroupMassType'][:,4] > 0, 
                    group_data['GroupMassType'][:,0] > 0,
                    (group_data['GroupMass']-group_data['GroupMassType'][:,1])/group_data['GroupMass'] < 0.15733)
group_filt = np.logical_and.reduce(group_conditions)

sub_idx = group_data['GroupFirstSub']
subhalo_conditions = (subhalo_data['SubhaloFlag'][sub_idx], subhalo_data['SubhaloMassType'][:,4][sub_idx] > 0, 
                      subhalo_data['SubhaloMassType'][:,0][sub_idx] > 0, sub_idx != -1)
filt = np.logical_and.reduce(subhalo_conditions + group_conditions)
group_ID = np.where(filt)[0]

# Define StellarMassExSitu and SubhaloBfldHalo
StellarMassExSitu = np.array(subhalo_data['StellarMassExSitu'])
SubhaloBfldHalo = subhalo_data['SubhaloBfldHalo'][sub_idx][filt]

quasar_energy = [np.sum(il.snapshot.loadHalo(basePath, 99, ID, 5, fields=['BH_CumEgyInjection_QM'])) for ID in group_ID]
quasar_energy_arr = np.array(quasar_energy)
quasar_energy_arr[group_data['GroupBHMass'][filt] == 0] = 0
quasar_energy_arr = quasar_energy_arr.astype('float32')

wind_energy = [np.sum(il.snapshot.loadHalo(basePath, 99, ID, 5, fields=['BH_CumEgyInjection_RM'])) for ID in group_ID]
wind_energy_arr = np.array(wind_energy)
wind_energy_arr[group_data['GroupBHMass'][filt] == 0] = 0
wind_energy_arr = wind_energy_arr.astype('float32')

SFG = []
for ID in group_ID:
    SFR = il.snapshot.loadHalo(basePath, 99, ID, 0, fields=['StarFormationRate'])
    Mass = il.snapshot.loadHalo(basePath, 99, ID, 0, fields=['Masses'])
    SFG.append(sum(Mass[SFR > 0]))

from colossus.cosmology import cosmology
cosmo=cosmology.setCosmology('planck15')
import numpy as np

# MST feature in Gyr (converted from scale factor)
MST = []

for ID in group_ID:
    SFT = il.snapshot.loadHalo('/home/kellerbw/illustris/TNG100-1', 99, ID, 4, fields=["GFM_StellarFormationTime"])
    SFT = (SFT[SFT > 0])  # Stellar Formation Time in scale factor
    z = 1.0 / SFT - 1.0  # Converting scale factor to redshift
    age_of_universe_at_z = np.mean(cosmo.age(z)) # Getting the age of the universe at that redshift in Gyr
    MST.append(age_of_universe_at_z)  # Storing the mean stellar formation time in Gyr


# Metal Fields
metalFields = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe']


data_dict = {}
data_dict['QuasarEnergy'] = 2.938e53 * np.array(quasar_energy_arr, dtype=np.float64)
data_dict['WindEnergy'] = 2.938e53 * np.array(wind_energy_arr, dtype=np.float64)

data_dict['SFG'] = M_unit*np.array(SFG)
data_dict['MST'] = np.array(MST)
#data_dict['StellarMassExSitu'] = M_unit*np.array(StellarMassExSitu)
#data_dict['SubhaloBfldHalo'] = 2.60*np.array(SubhaloBfldHalo)

for k in groupFields:
    if k == 'GroupFirstSub':
        continue
    if k == 'GroupMassType':
        data_dict['GroupGasMass'] = M_unit*group_data['GroupMassType'][:,0][filt]
        data_dict['GroupStarMass'] = M_unit*group_data['GroupMassType'][:,4][filt]
        data_dict['GroupDMMass'] = M_unit*group_data['GroupMassType'][:,1][filt]
        continue
    # field_data = data_dict[k]
    if k == 'GroupGasMetalFractions':
        for j in range(9):
            data_dict['GroupGas'+metalFields[j]+'Fraction'] = group_data['GroupGasMetalFractions'][:,j][filt]
        continue
    if k == 'GroupStarMetalFractions':
        for j in range(9):
            data_dict['GroupStar'+metalFields[j]+'Fraction'] = group_data['GroupStarMetalFractions'][:,j][filt]
        continue
    if 'Group_M_Crit200' in group_data:
            data_dict['Group_M_Crit200'] = group_data['Group_M_Crit200'][filt] * M_unit
        
            data_dict[k] = group_data[k][filt]
    if k in groupMassFields:
            data_dict[k] *= M_unit
    
for k in subhaloFields:
    if k == 'SubhaloFlag':
        continue
    if k == 'SubhaloMassType':
        data_dict['SubhaloGasMass'] = M_unit*subhalo_data['SubhaloMassType'][:,0][sub_idx][filt]
        data_dict['SubhaloStarMass'] = M_unit*subhalo_data['SubhaloMassType'][:,4][sub_idx][filt]
        data_dict['SubhaloDMMass'] = M_unit*subhalo_data['SubhaloMassType'][:,1][sub_idx][filt]
        continue
    if k == 'SubhaloMassInRadType':
        data_dict['SubhaloGasMassInRad'] = M_unit*subhalo_data['SubhaloMassInRadType'][:,0][sub_idx][filt]
        data_dict['SubhaloStarMassInRad'] = M_unit*subhalo_data['SubhaloMassInRadType'][:,4][sub_idx][filt]
        data_dict['SubhaloDMMassInRad'] = M_unit*subhalo_data['SubhaloMassInRadType'][:,1][sub_idx][filt]
        continue
    if k == 'SubhaloMassInHalfRadType':
        data_dict['SubhaloGasMassInHalfRad'] = M_unit*subhalo_data['SubhaloMassInHalfRadType'][:,0][sub_idx][filt]
        data_dict['SubhaloStarMassInHalfRad'] = M_unit*subhalo_data['SubhaloMassInHalfRadType'][:,4][sub_idx][filt]
        data_dict['SubhaloDMMassInHalfRad'] = M_unit*subhalo_data['SubhaloMassInHalfRadType'][:,1][sub_idx][filt]
        continue
    if k == 'SubhaloMassInMaxRadType':
        data_dict['SubhaloGasMassInMaxRad'] = M_unit*subhalo_data['SubhaloMassInMaxRadType'][:,0][sub_idx][filt]
        data_dict['SubhaloStarMassInMaxRad'] = M_unit*subhalo_data['SubhaloMassInMaxRadType'][:,4][sub_idx][filt]
        data_dict['SubhaloDMMassInMaxRad'] = M_unit*subhalo_data['SubhaloMassInMaxRadType'][:,1][sub_idx][filt]
        continue  
        
    if k == 'SubhaloHalfmassRadType':
        data_dict['SubhaloStarHalfmassRad'] = little_h * subhalo_data['SubhaloHalfmassRadType'][:, 4][sub_idx][filt]
        data_dict['SubhaloGasHalfmassRad'] = little_h * subhalo_data['SubhaloHalfmassRadType'][:, 0][sub_idx][filt]
        continue

    if k == 'SubhaloGasMetalFractions':
        for j in range(9):
            data_dict['SubhaloGas'+metalFields[j]+'Fraction'] = subhalo_data['SubhaloGasMetalFractions'][:,j][sub_idx][filt]
        continue
    if k == 'SubhaloStarMetalFractions':
        for j in range(9):
            data_dict['SubhaloStar'+metalFields[j]+'Fraction'] = subhalo_data['SubhaloStarMetalFractions'][:,j][sub_idx][filt]
        continue
        
    data_dict[k] = subhalo_data[k][sub_idx][filt] 
    if k in subhaloMassFields: 
        data_dict[k] *= M_unit

        

df100= pd.DataFrame(data=data_dict)


if 'SubhaloStarHalfmassRad' in df100.columns:
    print("SubhaloStarHalfmassRad exists in the DataFrame")
else:
    print("SubhaloStarHalfmassRad does not exist in the DataFrame")



if 'SubhaloStarHalfmassRad' in df100.columns:
    print("SubhaloStarHalfmassRad exists in the DataFrame")
else:
    print("SubhaloStarHalfmassRad does not exist in the DataFrame")

df100['StellarMassExSitu'] = df100['StellarMassExSitu'] * M_unit
df100['SubhaloBfldHalo'] = df100['SubhaloBfldHalo']*2.60e-6


little_h = 0.6774
M_unit = 1e10 / little_h  # Mass unit conversion factor

# Listed the features that need conversion
mass_features = ['SubhaloMass', 'SubhaloMassInRad', 'SubhaloMassInHalfRad', 'SubhaloMassInMaxRad']
radius_features = ['SubhaloHalfmassRad', 'SubhaloVmaxRad']

# Applying unit conversions for mass features
for feature in mass_features:
    if feature in df100.columns:
        df100[feature] = df100[feature] * M_unit

# Applying unit conversions for radius features
for feature in radius_features:
    if feature in df100.columns:
        df100[feature] = df100[feature] * little_h

# Adding SubhaloBfldDisk feature and applying its unit conversion (multiply by 2.60)
if 'SubhaloBfldDisk' in df100.columns:
    df100['SubhaloBfldDisk'] = df100['SubhaloBfldDisk'] * 2.60

df100.to_pickle("full_illustris_df100.pkl")
