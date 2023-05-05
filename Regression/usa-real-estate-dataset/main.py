# %%
import opendatasets as od
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

if not os.path.exists("../datasets"):
    os.makedirs("../datasets")

PROJECT_URL = "https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset"
PROJECT_NAME = PROJECT_URL.split("/")[-1]
DATA_PATH = os.path.join("../datasets", PROJECT_NAME)
od.download(PROJECT_URL, "../datasets")
# %%
df = pd.read_csv(os.path.join(DATA_PATH, "realtor-data.csv"))
# %%
df.info()
# Status seems irrelevant, prev_sold_date 71% missing values
df.drop(columns=["status", "prev_sold_date"], inplace=True)
# %% 
print(df.state.value_counts())
# Drop states with low sample size
print(df.shape)
df = df.loc[~df.state.isin(["South Carolina", "Tennessee", "Virginia", "New Jersey"]), :]
print(df.shape)
# %%
df.describe()
# From the above we can see we have a lot of outliers
# %%
df.describe(include='O').T
# %%
df.isnull().sum()
# %%
df.groupby('state').price.mean().sort_values(ascending=False).plot.bar()
plt.title('Average Price of Real Estate for US states')
plt.show()
df.groupby('city').price.mean().sort_values(ascending=False).head(20).plot.bar()
plt.title('Average Price per city')
plt.show()
# %%
# %% Trying to drop rows that have at least one null value
df_dropna_all = df.dropna()
# %% To verify that we still have outliers
sns.kdeplot(df_dropna_all["price"])
# %% I will consider an outlier everything that has z score above 3.5
def find_outliers(col):

    mean, std = col.describe()[["mean", "std"]]
    z_score = col.map(lambda x: np.abs((mean-x)/std))
    print(f"Found {(z_score > 3.5).sum()} outliers")

    return z_score > 3

print(df_dropna_all.shape)
df_dropna_all = df_dropna_all.loc[~find_outliers(df_dropna_all["price"]), :]
print(df_dropna_all.shape)
sns.scatterplot(x=df_dropna_all["acre_lot"], y=df_dropna_all["price"])
# %% Normalize the target
#df_dropna_all["price"] = stats.boxcox(df_dropna_all["price"])[0]
#sns.kdeplot(df_dropna_all["price"], color="green", fill=True)
# %%
#sns.pairplot(df_dropna_all[["bed", "bath", "acre_lot", "house_size", "price"]])
# %%
X_dropna_all = df_dropna_all.drop(columns=["price"])
y_dropna_all = df_dropna_all["price"].copy()

# %% First lets try only with numerical columns
X_dropna_all_num = df_dropna_all.select_dtypes(exclude=["object"])
X_train_dropna_all_num, X_test_dropna_all_num, y_train_dropna_all, y_test_dropna_all = train_test_split(X_dropna_all_num, y_dropna_all, test_size=0.2, random_state=47)


reg = RandomForestRegressor(n_jobs=-1, random_state=47)
reg.fit(X_train_dropna_all_num, y_train_dropna_all)
preds = reg.predict(X_test_dropna_all_num)
print(np.sqrt(mean_squared_error(preds, y_test_dropna_all)))
# %% Adding categorical columns
X_dropna_all_oh = pd.get_dummies(X_dropna_all)
X_train_dropna_all_oh, X_test_dropna_all_oh, y_train_dropna_oh, y_test_dropna_oh = train_test_split(X_dropna_all_oh, y_dropna_all, test_size=0.2, random_state=47)

reg = RandomForestRegressor(n_jobs=-1, random_state=47)
reg.fit(X_train_dropna_all_oh, y_train_dropna_oh)
preds = reg.predict(X_test_dropna_all_oh)
print(np.sqrt(mean_squared_error(preds, y_test_dropna_oh)))
# From the above looks like state and city add a lot of noise and hurt performance a lot
# %%
cat = CatBoostRegressor(random_state=47)
cat.fit(X_train_dropna_all_num, y_train_dropna_all)
preds = cat.predict(X_test_dropna_all_num)
print(np.sqrt(mean_squared_error(preds, y_test_dropna_all)))
# %%
rid = Ridge(random_state=47)
rid.fit(X_train_dropna_all_num, y_train_dropna_all)
preds = rid.predict(X_test_dropna_all_num)
print(round(np.sqrt(mean_squared_error(preds, y_test_dropna_all)), 5))
# %%
