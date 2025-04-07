# Baryon_Fraction_ML
## Description of Scripts
* `Final Preprocessing.ipynb`-A Jupyter notebook that loads raw data from the IllustrisTNG simulation, filters halos based on stellar mass and gas mass criteria, creates a cleaned Dataframe, and then processed data is saved as full_illustris_df100.pkl for further analysis.
* `gen_ebm_accuracy.py`-Generates a scatter plot comparing actual vs. predicted baryon fraction values along with the R² score for the EBM model.
* `gen_ebm_bistd_best.py`-Trains multiple EBM models, computes the standard deviation of bivariate interaction functions across models, and visualizes the two significant interaction functions to highlight the variability and model uncertainty.
* `gen_ebm_bivariate.py`-Loads a trained EBM model, selects two specific bivariate interaction functions, and visualizes their interaction functions as heatmaps.
* `gen_ebm_feat_imp_multiple.py`-Trains multiple EBM models, aggregates the univariate feature effects across models, and visualizes their mean and min-max range to highlight the stability of each feature importance.
* `gen_ebm_trained.py`-Fits an EBM model to the five most significant features selected from permutation importance, with hyperparameter tuning performed using GridSearchCV.
* `gen_ebm_univariate.py`-Loads a trained EBM model and visualizes the univariate feature importance for the top five most important features.
* `gen_rf_accuracy.py`-Visualizes the Random Forest model’s performance by plotting predicted baryon fractions against true values.
* `gen_rf_trained.py`-Trains a RF model on the training dataset using optimized hyperparameters.
* `gen_ustd.py`-Creates a boxplot of feature importances across 10 EBM models for the top five features. The importances are mean-centered and scaled to visualize variability.
* `gen_Z_score.py`-Trains multiple EBM models using the five most important features, calculates the Z-scores of bivariate interaction functions across these models, and visualizes the two most significant interactions.
* `preprocess_data.py`-Prepare the dataset for training the models, applying feature filtering, and performing log transformations.
* `Model_Training.ipynb`-A Jupyter notebook that trains an RF model, performs feature selection using permutation importance, and then trains an EBM model using the top five most significant features.
