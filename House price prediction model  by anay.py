# %%
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# Enable inline plotting
%matplotlib inline

# %% [markdown]
# ## Load Dataset
# %%
housing = pd.read_csv("data1.csv")

# %% [markdown]
# ## Exploratory Data Analysis
# %%
housing.head()

# %%
housing.info()

# %%
housing['CHAS'].value_counts()

# %%
housing.describe()

# %%
housing.hist(bins=50, figsize=(20, 15))

# %% [markdown]
# ## Train-Test Splitting

# %%
def split_train_test(data, test_ratio=0.2):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# %%
train_set, test_set = split_train_test(housing, 0.2)

# %%
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")

# %%
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")

# %%
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_test = housing.loc[train_index]
    strat_test_test = housing.loc[test_index]

# %%
housing = strat_train_test.copy()

# %% [markdown]
# ## Correlation Analysis
# %%
corr_matrix = housing.corr()

# %%
corr_matrix['MEDV'].sort_values(ascending=False)

# %% [markdown]
# ### Scatter Matrix
# %%
from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize=(12, 8))

# %%
housing.plot(kind="scatter", x='RM', y='MEDV', alpha=0.8)

# %% [markdown]
# ## Attribute Combinations
# %%
housing['taxRM'] = housing['TAX'] / housing['RM']

# %%
housing = strat_train_test.drop('MEDV', axis=1)
housing_labels = strat_train_test['MEDV'].copy()

# %%
imputer = SimpleImputer(strategy='median')
imputer.fit(housing)

# %%
X = imputer.transform(housing)
housing_ar = pd.DataFrame(X, columns=housing.columns)

# %% [markdown]
# ## Data Preprocessing Pipeline
# %%
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# %%
housing_ar_num = my_pipeline.fit_transform(housing)

# %% [markdown]
# ## Model Selection
# %%
mode = RandomForestRegressor()
mode.fit(housing_ar_num, housing_labels)

# %%
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)

# %%
mode.predict(prepared_data)

# %% [markdown]
# ## Model Evaluation
# %%
housing_predictions = mode.predict(housing_ar_num)
line_mse = mean_squared_error(housing_labels, housing_predictions)
line_rmse = np.sqrt(line_mse)

# %%
scores = cross_val_score(mode, housing_ar_num, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

# %%
def print_scores(scores):
    print("Scores:", scores)
    print("Mean value:", scores.mean())
    print("Standard deviation:", scores.std())

# %%
print_scores(rmse_scores)

# %% [markdown]
# ## Save the Model
# %%
dump(mode, 'ML_project_1_model.joblib')

# %% [markdown]
# ## Testing the Model
# %%
X_test = strat_test_test.drop('MEDV', axis=1)
Y_test = strat_test_test['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_prediction = mode.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_prediction)
final_rmse = np.sqrt(final_mse)

# %%
print("Final RMSE:", final_rmse)

# %%
plt.figure(figsize=(12, 8))
plt.plot(Y_test.values, label='Actual MEDV', color='skyblue')
plt.scatter(range(len(final_prediction)), final_prediction, color='black', alpha=0.8, label='Predicted MEDV')
plt.xlabel('Index')
plt.ylabel('House Price (MEDV)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid()
plt.show()

# user can use joblib load to run the program on their data
