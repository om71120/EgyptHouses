import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import missingno

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier


TRAIN_DATA_PATH = os.path.join(os.getcwd(),"Egypt_Houses_Price.csv")
df = pd.read_csv(TRAIN_DATA_PATH)
df.head()


df = df.drop(columns=["Compound", "Payment_Option", "Delivery_Date", "Delivery_Term", "Level"])

df = df.dropna()

df = df.drop_duplicates()

# Reset the index
df = df.reset_index(drop=True)


Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1

# Define outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["Price"] < lower_bound) | (df["Price"] > upper_bound)]


# Calculate the median of the column
median_value = df["Price"].median()

# Replace outliers with the median
df["Price"] = df["Price"].apply(
    lambda x: median_value if x < lower_bound or x > upper_bound else x
)

Q1 = df["Bedrooms"].quantile(0.25)
Q3 = df["Bedrooms"].quantile(0.75)
IQR = Q3 - Q1

# Define outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["Bedrooms"] < lower_bound) | (df["Bedrooms"] > upper_bound)]

median_value = df["Bedrooms"].median()

# Replace outliers with the median
df["Bedrooms"] = df["Bedrooms"].apply(
    lambda x: median_value if x < lower_bound or x > upper_bound else x
)
Q1 = df["Bathrooms"].quantile(0.25)
Q3 = df["Bathrooms"].quantile(0.75)
IQR = Q3 - Q1

# Define outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["Bathrooms"] < lower_bound) | (df["Bathrooms"] > upper_bound)]


median_value = df["Bathrooms"].median()

# Replace outliers with the median
df["Bathrooms"] = df["Bathrooms"].apply(
    lambda x: median_value if x < lower_bound or x > upper_bound else x
)


df["Furnished"] = df["Furnished"].replace({"Yes": 1, "No": 0, "Unknown": -1})

df = df.drop(columns="City")

x = df.drop(columns="Furnished", axis=1)
y = df["Furnished"]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=47
)


num_col = x_train.select_dtypes(include="number").columns.tolist()
categ_col = x_train.select_dtypes(exclude="number").columns.tolist()


num_pipe = Pipeline(
    steps=[
        ("selector", DataFrameSelector(num_col)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]
)
categ_pipe = Pipeline(
    steps=[
        ("selector", DataFrameSelector(categ_col)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder()),
    ]
)
all_pipe = FeatureUnion(
    transformer_list=[("categorical", categ_pipe), ("numerical", num_pipe)]
)


x_train_final = all_pipe.fit_transform(x_train)
x_test_finall = all_pipe.transform(x_test)

def house_process(x_new):

    df_new=pd.DataFrame([x_new])
    df_new.columns = x_train.column

    df_new["Price"] = df_new["Price"].astype("float64")
    df_new["Type"] = df_new["Type"].astype("str")
    df_new["Bedrooms"] = df_new["Bedrooms"].astype("int")
    df_new["Bathrooms"] = df_new["Bathrooms"].astype("int")
    df_new["Area"] = df_new["Area"].astype("int")

    x_process = all_pipe.transform(df_new)

    return x_process
