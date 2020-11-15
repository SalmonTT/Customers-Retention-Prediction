from this import s

import variable as variable
from scipy import stats
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3 import traceplot
import seaborn as sns
import pandas as pd
from theano import shared
from sklearn import preprocessing

def blr(X_train, X_test, y_train, y_test):
    # Formula for Bayesian Linear Regression
    formula = ‘UWC~ ‘ + ‘ + ‘.join([‘ % s’ % variable for variable in X_train.columns[1:]])
    print(formula)

    # Context for the model
    with pm.Model() as normal_model:
        # The prior for the model parameters will be a normal distribution
        family = pm.glm.families.Normal()
        # Creating the model requires a formula and data (and optionally a family)
        pm.GLM.from_formula(formula, data=X_train, family=family)
        # Perform Markov Chain Monte Carlo sampling
        normal_trace = pm.sample(draws=2000, chains=2, tune=500)


    # Define a function to calculate MAE and RMSE
    def evaluate_prediction(prediction, true):
        mae = np.mean(abs(predictions - true))
        rmse = np.sqrt(np.mean((predictions - true) ** 2))

        return mae, rmse


    median_pred = X_train['UWC'].median()
    median_preds = [median_pred for _ in range(len(X_test))]
    true = X_test['UWC']
    # Display mae and rmse
    mae, rmse = evaluate_prediction(median_preds, true)
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))