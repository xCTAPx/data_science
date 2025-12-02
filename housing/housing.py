import os
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from transformers import CombineAttributesAdder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.svm import SVR
from scipy.stats import expon, reciprocal

# <---load data --->
HOUSING_PATH = os.path.join('datasets', 'housing')

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

# <---create dataframe from loaded data & some analyze --->
housing_df = load_housing_data()
# print(housing_df.describe())

# housing_df.hist(bins=100, figsize=(20, 15))
# plot.show()

# <--- transform non-numeric attributes to numeric (will be later in pipeline) --->
# housing_cat = housing_df[['ocean_proximity']]
# print(housing_cat)
# cat_encoder = OneHotEncoder()
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# print(housing_cat_1hot)
# print(cat_encoder.categories_)

# Add one-hot encoded columns to housing_df
# housing_cat_1hot_df = pd.DataFrame(housing_cat_1hot.toarray(), 
#                                     columns=cat_encoder.get_feature_names_out(['ocean_proximity']),
#                                     index=housing_df.index)
# housing_df = pd.concat([housing_df.drop('ocean_proximity', axis=1), housing_cat_1hot_df], axis=1)

# <---splitting dataset to train & test sets --->

# train_set, test_set = train_test_split(housing_df, test_size=0.2, random_state=42)
# print(f"Train set: {train_set}, Test set: {test_set}")

housing_df['income_cat'] = pd.cut(housing_df['median_income'], # categories by income for creating stratified test set
                                  bins=[0, 1.5, 3, 4.5, 6, np.inf],
                                  labels=[1, 2, 3 ,4 ,5])

# housing_df['income_cat'].hist()
# plot.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(housing_df, housing_df['income_cat']):
    strat_train_set = housing_df.loc[train_idx]
    strat_test_set = housing_df.loc[test_idx]

t = strat_test_set['income_cat'].value_counts() / len(strat_test_set)

# print(t) # check proportions

# <--- delete excessive attributes (income_cat was created only for stratified splitting) --->
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

# <--- research data & their correlations --->
housing = strat_train_set.copy()
housing_without_str_attrs = housing.drop('ocean_proximity', axis=1)
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
              s=housing['population']/100, label='population',
              c='median_house_value', cmap=plot.get_cmap('jet'), colorbar=True
            )

# plot.legend()
# plot.show()

# print(housing)

corr_matrix = housing_without_str_attrs.corr()
visual_cm = corr_matrix['median_house_value'].sort_values(ascending=False)
# print(visual_cm)

attrs = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing_without_str_attrs[attrs])
# plot.show()

# <--- create some new useful attributes for model & research their --->
# housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
# housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
# housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing_without_str_attrs.corr()
visual_cm_2 = corr_matrix['median_house_value'].sort_values(ascending=False)
# print(visual_cm_2)

# <--- extract labels from dataset for train set to separate variable --->
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
# print(housing_labels)

# <--- fill empty attribute fields with median value --->
# imputer = SimpleImputer(strategy='median')
# imputer.fit(housing)

# X = imputer.transform(housing)
# housing_tr = pd.DataFrame(X, columns=housing.columns, index=housing.index) # transform back to pandas dataFrame

# <--- add combine attributes to dataset via transformer --->
# attrs_adder = CombineAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attrs_adder.transform(housing.values)
# print(housing_extra_attribs)

# <--- creating piplines --->
housing_num = housing.columns.drop('ocean_proximity')

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attrs_adder', CombineAttributesAdder()),
    ('std_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)

# <--- try LinearRegression model --->
lin_reg = LinearRegression()
print('Linear Regression train...')
lin_reg.fit(housing_prepared, housing_labels)
print('Linear Regression has been trained!')

some_data = housing.iloc[:5]
some_labels = housing_labels[:5]
some_data_prepared = full_pipeline.transform(some_data)
# print('Predictions:', lin_reg.predict(some_data_prepared))
# print('Labels:', list(some_labels))
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)

joblib.dump(lin_reg, './models/hosung_linear_regression.pkl')

# <--- try DecisionTreeRegressor model --->
tree_reg = DecisionTreeRegressor()
print('Decision Tree Regressor train...')
tree_reg.fit(housing_prepared, housing_labels)
print('Decision Tree Regressor has been trained!')
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)

joblib.dump(lin_reg, './models/hosung_decision_tree_regressor.pkl')

# <--- try K-fold cross-validation for comparing linear regression and decision tree models --->
scores = cross_val_score(
    tree_reg,
    housing_prepared,
    housing_labels,
    scoring='neg_mean_squared_error',
    cv=10
)

tree_rmse_scores = np.sqrt(-scores)

# print("Sum of estimations [tree]:", tree_rmse_scores)
# print("Mean [tree]:", tree_rmse_scores.mean())
# print("Standard deviation [tree]:", tree_rmse_scores.std())

lin_scores = cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring='neg_mean_squared_error',
    cv=10
)

lin_rmse_scores = np.sqrt(-lin_scores)

# print("Sum of estimations [lr]:", lin_rmse_scores)
# print("Mean [lr]:", lin_rmse_scores.mean())
# print("Standard deviation [lr]:", lin_rmse_scores.std())

# <--- try RandomForestRegressor --->
forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# forest_reg_predictions = forest_reg.predict(housing_prepared)
# forest_reg_mse = mean_squared_error(housing_labels, forest_reg_predictions)
# forest_reg_rmse = np.sqrt(forest_reg_mse)
# # print(forest_reg_rmse)

# forest_reg_scores = cross_val_score(
#     forest_reg,
#     housing_prepared,
#     housing_labels,
#     scoring='neg_mean_squared_error',
#     cv=10
# )

# forest_reg_rmse_scores = np.sqrt(-forest_reg_scores)

# print("Sum of estimations [forest_reg]:", forest_reg_rmse_scores)
# print("Mean [forest_reg]:", forest_reg_rmse_scores.mean())
# print("Standard deviation [forest_reg]:", forest_reg_rmse_scores.std())

joblib.dump(lin_reg, './models/hosung_random_forest_regressor.pkl')

# <--- search best model hyperparameters with GridSearchCV --->
param_grid = [
    {'n_estimators': [10, 20, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

print('Random Forest Regressor train via grid search...')
grid_search.fit(housing_prepared, housing_labels)
print('Random Forest Regressor has trained!')

# print(grid_search.best_params_)

# <--- importance of every attribute (feature) --->
feature_importances = grid_search.best_estimator_.feature_importances_

# print(feature_importances)

extra_attributes = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attrs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attributes + cat_one_hot_attrs
# print(sorted(zip(feature_importances, attributes), reverse=True))

# <--- estimate final model via test set --->
final_model = grid_search.best_estimator_

test_set = strat_test_set.drop('median_house_value', axis=1)
test_labels = strat_test_set['median_house_value'].copy()

test_set_prepared = full_pipeline.transform(test_set)
final_predictions = final_model.predict(test_set_prepared)

final_mse = mean_squared_error(test_labels, final_predictions)
final_rmse = np.sqrt(final_mse)

# print(final_rmse)

# <--- compute confidence for error --->
confidence = 0.95
squared_errors = (final_predictions - test_labels) ** 2
res = np.sqrt(stats.t.interval(
    confidence,
    len(squared_errors) - 1,
    loc=squared_errors.mean(),
    scale=stats.sem(squared_errors)
))

# print(res)

# <--- try other models (for example, SVR) --->
svr = SVR()
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

rnd_search_svr = RandomizedSearchCV(svr, param_distribs,
                                    n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                    verbose=2, random_state=42)

print('Support Vector Regressor train via RandomizedSearchCV...')
rnd_search_svr.fit(housing_prepared, housing_labels)
print('Support Vector Regressor has been trained!')
print('Best params:', rnd_search_svr.best_params_)
print('Best rmse:', np.sqrt(-rnd_search_svr.best_score_))

joblib.dump(lin_reg, './models/hosung_svr.pkl')