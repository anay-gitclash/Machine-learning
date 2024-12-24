# %%
import pandas as pd 
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from joblib import dump, load
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

%matplotlib inline

# %%
housing = pd.read_csv("data1.csv")

# %%
housing.head()

# %%
housing.info()

# %%
housing['CHAS'].value_counts()

# %%
housing.describe()

# %%
housing.hist(bins=50 , figsize=[20,15])


# %% [markdown]
# ## train-test splitting

# %%
# for leanring perpose only 
def splt_train_test(data = housing , test_ratio= 0.2):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size= int(len(data)* test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# %%
train_set , test_set= splt_train_test(housing,0.2)

# %%
print(f"rows in train set: {len(train_set)}\nROws in test set:{len(test_set)}")

# %%
train_set , test_set = train_test_split(housing, test_size=0.2,random_state=42)
print(f"rows in train set: {len(train_set)}\nROws in test set:{len(test_set)}")

# %%
housing['CHAS'].value_counts()

# %%
slpit = StratifiedShuffleSplit(n_splits=1 , test_size=0.2 , random_state=42)
for train_index , test_index in slpit.split(housing, housing['CHAS']):
    strat_train_test = housing.loc[train_index]
    strat_test_test =  housing.loc[test_index]

# %%
strat_test_test['CHAS'].value_counts()

# %%
housing = strat_train_test.copy()

# %% [markdown]
# ## Looking for correlation

# %%
corr_matrix= housing.corr()

# %%
corr_matrix['MEDV'].sort_values(ascending=False)

# %%
from pandas.plotting import scatter_matrix

# %%
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix( housing[attributes], figsize=(12,8))


# %%
housing.plot(kind="scatter",x='RM', y='MEDV',alpha = 0.8)

# %% [markdown]
# ## Trying out attribute combinations

# %%
housing['taxRM'] = housing['TAX']/housing['RM']

# %%
housing = strat_train_test.drop('MEDV', axis=1)
housing_labels = strat_train_test['MEDV'].copy()

# %%
#housing.shape

# %%
imputer = SimpleImputer(strategy='median')
imputer.fit(housing)

# %%
X = imputer.transform(housing)

# %%
housing_ar = pd.DataFrame(X, columns=housing.columns)

# %%
#housing_ar.describe()

# %% [markdown]
# ## Creating a pipeline

# %% [markdown]
# ### feature scaling

# %%
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])

# %%
housing_ar_num = my_pipeline.fit_transform(housing)


# %%
#housing_ar_num

# %% [markdown]
# ## selecting desired model for projet

# %%
#mode = DecisionTreeRegressor()
mode = RandomForestRegressor()
mode.fit(housing_ar_num,housing_labels)
             

# %%
some_data = housing.iloc[:5]
#some_data

# %%
some_labels = housing_labels.iloc[:5]
#some_labels

# %%
prepared_data = my_pipeline.transform(some_data)
#prepared_data

# %%
mode.predict(prepared_data)

# %%
#list(some_labels)

# %%
housing_predictions = mode.predict(housing_ar_num)
line_mse = mean_squared_error(housing_labels, housing_predictions)
line_rmse = np.sqrt(line_mse)



# %%
#line_mse

# %%
scores = cross_val_score(mode, housing_ar_num, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

# %%
#rmse_scores

# %%
def print_scores(scores):
    print("scores: ", scores)
    print("mean value: ", scores.mean())
    print("standard daviation ", scores.std())


# %%
#print_scores(rmse_scores)

# %% [markdown]
# ## save the model

# %%
dump(mode ,'ML project 1 by anay' )

# %% [markdown]
# ## testing the model

# %%
X_test = strat_test_test.drop('MEDV', axis= 1)
Y_test = strat_test_test['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_prediction = mode.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_prediction)
final_rmse = np.sqrt(final_mse)
#print(final_prediction)

# %%
final_rmse

# %%
plt.figure(figsize=(12,8))
plt.plot(Y_test.values, label = 'Actual MEDV',color = 'skyblue')
plt.xlabel('Value index')
plt.ylabel('house price(MEDV)')
plt.title('Actual vs predicted house prices')
plt.scatter(range(len(final_prediction)),final_prediction,color = 'Black', alpha= 0.8,label = 'predicted values')
plt.legend()

# %%
#mode = load('ML project 1 by anay')
# %%
#mode.predict(prepared_data)
# %%
#features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       #-11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       #-0.97491834,  0.41164221, -66.86091034]])
#mode.predict(features)
# User can give data and use the model as above
# %%
