DATASET_PATH = "datasets/full.csv"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open(DATASET_PATH) as f:
    # load labels (first line) manually
    label = f.readline().replace("\n","").split(",")[:-1]
    # count number of parameters, column count is this plus 1 (for label)
    # therefore, label (last column) has this index
    last_column_index = len(label)

# read with pandas for autodetect of mixed datatypes (int, flaot and string)
df = pd.read_csv(DATASET_PATH, sep=',', usecols=range(0, last_column_index+1))
target = df["Risk_Flag"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# all str_cols are actually containing enumerating strings
str_cols = [
    "Married/Single",
    "House_Ownership",
    "Profession",
    "CITY",
    "STATE"
]

# mark str_cols as type category
X = df.drop(columns="Risk_Flag")
for i in str_cols:
    X[i] = X[i].astype("category")

# translate yes/no col to boolean value
yesno_cols = ["Car_Ownership"]
X[yesno_cols] = X[yesno_cols].replace({"yes": "1", "no": "0"}).astype("int")

y = target

# split data/target into test and training vectors
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
print("data     ", X.shape)
print("train set", X_train.shape, y_train.shape)
print("test  set", X_test.shape, y_test.shape)

# one-hot-encode X. This is better than enumeration, because it removes all correlation between the different values of one column.
enc = make_column_transformer(
        (OneHotEncoder(sparse_output=False), X.dtypes == 'category'),
        remainder='passthrough', verbose_feature_names_out=False)
# teach encoder with complete frame
enc.fit(X)
# apply encoder to split data
X_train = pd.DataFrame(enc.transform(X_train), columns=enc.get_feature_names_out(), index=X_train.index)
X_test  = pd.DataFrame(enc.transform(X_test ), columns=enc.get_feature_names_out(), index=X_test .index)

# run simple classifier from assgmt 2
clf = LogisticRegression(solver="lbfgs", max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("accuracy", accuracy_score(y_pred, y_test))

# y_pred: predicted results
# y_test: actual (expected) results
