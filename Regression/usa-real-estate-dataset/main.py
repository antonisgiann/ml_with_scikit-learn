# %%
import opendatasets as od
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score


if not os.path.exists("../datasets"):
    os.makedirs("../datasets")

PROJECT_URL = "https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset"
PROJECT_NAME = PROJECT_URL.split("/")[-1]
DATA_PATH = os.path.join("../datasets", PROJECT_NAME)
od.download(PROJECT_URL, "../datasets")
# %%
df = pd.read_csv(os.path.join(DATA_PATH, "realtor-data.csv"))
df.head()
# %%
df.info()
# %%
df.describe()
# From the above we can see we have a lot of outliers
# %%
df.describe(include='O').T
# %%
df.isnull().sum()
# Status seems irrelevant , prev_sold_date 71% missing values
df.drop(columns=["status", "prev_sold_date"], inplace=True)
# %% 
print(df.state.value_counts())
# Drop states with low sample size
print(df.shape)
df = df.loc[~df.state.isin(["South Carolina", "Tennessee", "Virginia", "New Jersey"]), :]
print(df.shape)
# %%
df.groupby('state').price.mean().sort_values(ascending=False).plot.bar()
plt.title('Average Price of Real Estate for US states')
plt.show()
df.groupby('city').price.mean().sort_values(ascending=False).head(20).plot.bar()
plt.title('Average Price per city')
plt.show()
# %%
# %% Drop rows that have at least one null value
df_dropna_all = df.dropna()
# %% From the output of df.describe() we see that we have outliars. I will remove them using the rule anything outside 
# the interval [Q1 - interquartile range * 1.5,  Q3 + interquartile range * 1.5] is an outliar.
# Running this in an iterative manner for every column gives better performance compared to precalculating
# them for each column before removing the outliars.
def find_outliers(col):

    Q1, Q3 = col.describe()[["25%", "75%"]]
    IQR = Q3 - Q1
    high = Q3 + IQR * 1.5
    low = Q1 - IQR * 1.5
    is_outliar = col.map(lambda x: x > high or x < low)
    print(f"Found {(is_outliar).sum()} outliers")

    return is_outliar

print(df_dropna_all.shape)
for c in ["bed", "bath", "acre_lot", "house_size", "price"]:
    print(f"checking {c}")
    
    df_dropna_all = df_dropna_all.loc[~find_outliers(df_dropna_all[c]), :]
print(df_dropna_all.shape)

df_dropna_all["bed"] = df_dropna_all["bed"].astype(int)
df_dropna_all["bath"] = df_dropna_all["bath"].astype(int)
df_dropna_all["house_size"] = df_dropna_all["house_size"].astype(int)
# %%
sns.scatterplot(df_dropna_all, x="acre_lot", y="price", hue="state", alpha=0.1)
plt.show()
sns.scatterplot(df_dropna_all, x="house_size", y="price", hue="state", alpha=0.1)
plt.show()
corr_mat = df_dropna_all[["house_size", "bed", "acre_lot", "bath", "price"]].corr()
mask = np.triu(np.ones_like(corr_mat)).astype(bool)
mask = mask[1:, :-1]
corr_mat = corr_mat.iloc[1:,:-1]
sns.heatmap(corr_mat, annot=True, linewidth=1.5, fmt=".2f", cmap="crest", mask=mask)
plt.show()
sns.kdeplot(df_dropna_all["price"], fill=True, color="#0092A0")
plt.show()
sns.boxplot(df_dropna_all, x="state", y="price")
plt.xticks(rotation=45)
plt.show()
sns.boxplot(df_dropna_all, x="bed", y="price")
plt.show()
sns.boxplot(df_dropna_all, x="bath", y="price")
plt.show()
# %% Normalize the target
#df_dropna_all["price"] = stats.boxcox(df_dropna_all["price"])[0]
#sns.kdeplot(df_dropna_all["price"], color="green", fill=True)
# %%
X_dropna_all = df_dropna_all.drop(columns=["price"])
y_dropna_all = df_dropna_all["price"].copy()
##############
####MODELS####
##############
# %% First lets try only with numerical columns
X_dropna_all_num = X_dropna_all.select_dtypes(exclude=["object"])
X_train_dropna_all_num, X_test_dropna_all_num, y_train_dropna_all, y_test_dropna_all = train_test_split(X_dropna_all_num, y_dropna_all, test_size=0.2, random_state=47)

forest_reg = RandomForestRegressor(n_jobs=-1, random_state=47)
score = np.sqrt(-cross_val_score(forest_reg, X_train_dropna_all_num, y_train_dropna_all, scoring="neg_mean_squared_error", cv=5, n_jobs=-1))
print("Cross validation Mean squared error:", round(np.mean(score), 5))
# %% Ridge regression
l2_reg = Ridge(random_state=47)
score = np.sqrt(-cross_val_score(l2_reg, X_train_dropna_all_num, y_train_dropna_all, scoring="neg_mean_squared_error", cv=5, n_jobs=-1))
print("Cross validation Mean squared error:", round(np.mean(score), 5))
# %% Catboost
cat = CatBoostRegressor(random_state=47, verbose=0)
score = np.sqrt(-cross_val_score(cat, X_train_dropna_all_num, y_train_dropna_all, scoring="neg_mean_squared_error", cv=5, n_jobs=-1))
print("Cross validation Mean squared error:", round(np.mean(score), 5))
# %% Adding categorical columns to check performance
X_dropna_all_oh = pd.get_dummies(X_dropna_all)
X_train_dropna_all_oh, X_test_dropna_all_oh, y_train_dropna_oh, y_test_dropna_oh = train_test_split(X_dropna_all_oh, y_dropna_all, test_size=0.2, random_state=47)

# Random forest
forest_reg = RandomForestRegressor(n_jobs=-1, random_state=47)
score = np.sqrt(-cross_val_score(forest_reg, X_train_dropna_all_oh, y_train_dropna_oh, scoring="neg_mean_squared_error", cv=5, n_jobs=-1))
print("Cross validation Mean squared error:", round(np.mean(score), 5))
# %% Ridge regression
l2_reg = Ridge(random_state=47)
score = np.sqrt(-cross_val_score(l2_reg, X_train_dropna_all_oh, y_train_dropna_oh, scoring="neg_mean_squared_error", cv=5, n_jobs=-1))
print("Cross validation Mean squared error:", round(np.mean(score), 5))
# %%
