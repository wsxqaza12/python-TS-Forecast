import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_absolute_percentage_error
import catboost as cb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from chi_function import *
pd.options.mode.chained_assignment = None

# ===================================
# ========== 1. Setup Env. ==========
# ===================================
SEED = 1234
FOLDER = "result"
data = pd.read_csv('pltv_data.csv')

if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)


# =======================================
# ========== 2. Pre-processing ==========
# =======================================
col_keep = ['campaign', 'publisher', 'date', 'installs', 'd90_arpu']
data = data[col_keep]
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

category = ['campaign', 'publisher']
numeric = list(set(data.columns) - set(category))
for colname in category:
    data[colname] = data[colname].astype('str')
    
data_dropna = data.dropna(subset=['d90_arpu'])


# ================================================
# ========== 3.Find Best Hyperparameter ==========
# ================================================
# 1. Setup Backtesting Fold and Interval ############
backtesting_times = 3
test_time_range = split_backtest(data_dropna, time=backtesting_times, day=10)

# 2. Using Grid Search to Optimise Parameters #######
params = {'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10],
          'iterations': [250, 100, 500, 1000],
          'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]}
params_df = expand_grid(params)

# 3. Combining Backtesting with Grid Search #########
for i in range(len(params_df)):
    for k in range(backtesting_times):
        test_start = test_time_range[0][k]
        test_end = test_time_range[1][k]
        X_train, X_test, y_train, y_test = split_X_y_by_time(
            data_dropna, test_start, test_end)
        categorical_indicies = get_categorical_indicies(X_train)

        train_dataset = cb.Pool(
            X_train, y_train, cat_features=categorical_indicies)
        test_dataset = cb.Pool(
            X_test, y_test, cat_features=categorical_indicies)

        model = cb.CatBoostRegressor(iterations=params_df.loc[i]['iterations'],
                                     learning_rate=params_df.loc[i]['learning_rate'],
                                     depth=params_df.loc[i]['depth'])

        model.fit(train_dataset)
        y_predict = model.predict(test_dataset)
        MAPE_vaule = mean_absolute_percentage_error(
            y_true=y_test['d90_arpu'], y_pred=y_predict)
        params_df.loc[params_df.index[i], ('test' + str(k))] = MAPE_vaule

params_df['mean_MAPE'] = (
    params_df.test0 + params_df.test1 + params_df.test2)/3  # if only 3
best_Hyperparameter = params_df.sort_values(by=['mean_MAPE']).iloc[1]

# =========================================
# ========== 4.Training All Data ==========
# =========================================
X, y = split_X_y(data_dropna, date_col='date', y_col=['d90_arpu'])
train_dataset = cb.Pool(X, y, cat_features=categorical_indicies)

model = cb.CatBoostRegressor(iterations=best_Hyperparameter['iterations'],
                             learning_rate=best_Hyperparameter['learning_rate'],
                             depth=best_Hyperparameter['depth'])
model.fit(train_dataset)
print(f'Backtesting MAPE:{round(best_Hyperparameter.mean_MAPE, 3)}')

# ===============================
# ========== 5.Predict ==========
# ===============================
data_predict_X, data_predict_y = split_X_y(
    data, date_col='date', y_col=['d90_arpu'])
predict_train = cb.Pool(data_predict_X, data_predict_y,
                        cat_features=categorical_indicies)

data_to_csv = data[col_keep]
data_to_csv['d90_arpu_pred'] = model.predict(predict_train)
data_to_csv.to_csv(os.path.join(FOLDER, "predict.csv"), index=False)

# ============================
# ========== 6.Plot ==========
# ============================
# 1. Feature importance #######
feature_importanceg = plot_feature_importance(
    model.get_feature_importance(), X.columns, 'CATBOOST')
feature_importanceg.savefig(os.path.join(FOLDER, "feature_importance.png"))

# 2. shap #####################
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# 2.1 beeswarm ------------------------
fig = plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.savefig(os.path.join(FOLDER, 'beeswarm.png'), bbox_inches='tight')

# 2.2 plots_bar ------------------------
fig = plt.figure()
shap.plots.bar(shap_values, show=False)
plt.savefig(os.path.join(FOLDER, 'plots_bar.png'), bbox_inches='tight')

# 2.3 decision_plot --------------------
expected_values = explainer.expected_value
shap_array = explainer.shap_values(X)
fig = plt.figure()
shap.decision_plot(
    expected_values, shap_array[320:370], feature_names=list(X.columns), show=False)
plt.savefig(os.path.join(FOLDER, 'decision_plot.png'), bbox_inches='tight')
