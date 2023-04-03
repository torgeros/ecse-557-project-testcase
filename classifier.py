import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.neural_network import *
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score

def run(DATASET_PATH, onehot_cols, enum_cols, yesno_cols, target_col, drop_cols = [], should_print=False):
    """
    DATASET_PATH: string:    location of csv file to load
    onehot_cols:  list(str): column names that contain str/enum values to be encoded
    enum_cols:    list(str): column names that contain str/enum values to be encoded
    yesno_cols:   list(str): column names that contain yes/no boolean values
    target_col:   string:    target column name
    should_print: bool:      should print process details
    """

    with open(DATASET_PATH) as f:
        # load labels (first line) manually
        label = f.readline().replace("\n","").split(",")[:-1]
        # count number of parameters, column count is this plus 1 (for label)
        # therefore, label (last column) has this index
        last_column_index = len(label)
    if should_print:
        print("labels:", label)

    # read with pandas for autodetect of mixed datatypes (int, flaot and string)
    df = pd.read_csv(DATASET_PATH, sep=',', usecols=range(0, last_column_index+1))

    # drop elements of drop_cols
    df.drop(columns=drop_cols)
    for l in label:
        if l in drop_cols:
            label.remove(l)

    target = df[target_col]
    X = df.drop(columns=target_col)

    # mark string columns as type category
    for i in onehot_cols + enum_cols:
        X[i] = X[i].astype("category")

    X[yesno_cols] = X[yesno_cols].replace({"yes": "1", "no": "0"}).astype("int")

    y = target

    # split data/target into test and training vectors
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
    if should_print:
        print("data     ", X.shape)
        print("train set", X_train.shape, y_train.shape)
        print("test  set", X_test.shape, y_test.shape)

    # encode X
    enc = make_column_transformer(
            (OneHotEncoder(sparse_output=False), onehot_cols),
            (OrdinalEncoder(encoded_missing_value=0), enum_cols),
            remainder='passthrough',
            verbose_feature_names_out=False)
    # teach encoder with complete frame
    enc.fit(X)
    # apply encoder to split data
    X_train = pd.DataFrame(enc.transform(X_train), columns=enc.get_feature_names_out(), index=X_train.index)
    X_test  = pd.DataFrame(enc.transform(X_test ), columns=enc.get_feature_names_out(), index=X_test .index)

    clf = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(5, 2), activation='relu', random_state=1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_pred, y_test)
    if should_print:
        print("accuracy", accuracy)

    # y_pred: predicted results
    # y_test: actual (expected) results

    return {
        "y_pred": y_pred,
        "y_test": y_test,
        "accuracy": accuracy
    }
