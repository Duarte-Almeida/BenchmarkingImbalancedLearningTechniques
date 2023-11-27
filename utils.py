import os
import random

import numpy as np
import pandas as pd
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# dict the handle various datasets, the list contains [fetch_link, handle_name, file_to_fetch, target_column_name]
datasets = {
    "baf" : ["sgpjesus/bank-account-fraud-dataset-neurips-2022", "baf", "Base.csv", "fraud_bool"]
}


def fetch_data(dataset):
    """
    Loads the dataset from kaggle, performing a 80-20 train-test split

    dataset: the name of the dataset (accepted: "baf")
    """
    assert dataset in datasets

    link = datasets[dataset][0]
    dir_name = datasets[dataset][1]
    dataset_file = datasets[dataset][2]
    target_name = datasets[dataset][3]

    if not os.path.exists("datasets/"):
        os.makedirs("datasets/")

    if not os.path.exists(f"datasets/{dir_name}"):
        os.makedirs(f"datasets/{dir_name}")

    if not os.path.exists(f"datasets/{dir_name}/{dataset_file}"):
        print(f"Dataset {dir_name} does not exist. Fetching dataset {dir_name}...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(link, dataset_file, f"datasets/{dir_name}")
        zf = ZipFile(f'datasets/{dir_name}/{dataset_file}.zip')
        zf.extractall(f'datasets/{dir_name}')
        zf.close()
        os.remove(f"datasets/{dir_name}/{dataset_file}.zip")

    df = pd.read_csv(f"datasets/{dir_name}/{dataset_file}")
    labelencoder = LabelEncoder()
    cat_feats = df.select_dtypes(include="object").columns

    for col in cat_feats:
        df[col] = labelencoder.fit_transform(df[col])

    for col in cat_feats:
        df[col] = df[col].astype('int')

    X = df.drop(columns=target_name)
    y = df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return {"train": (X_train, y_train),
            "test": (X_test, y_test), 
            "cat_feats": cat_feats.to_list()}
