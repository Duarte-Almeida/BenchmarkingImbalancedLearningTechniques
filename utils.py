import os, random, numpy as np, pandas as pd, time
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import pickle
import sys
from datasets import BAFHandler, IEEEHandler, MLGHandler, SparkovHandler

datasets = {"baf": BAFHandler(),
            "ieee": IEEEHandler(),
            "mlg": MLGHandler(),
            "sparkov": SparkovHandler()}

orders = {
    "preprocessing": ["Base", "Oversamplers", "Undersamplers", "Hybrid"],
    "inprocessing": ["Losses", "Ensembles"]
}

def fetch_data(dataset):
    """
    Loads the dataset from kaggle, performing a 80-20 train-test split

    dataset: the name of the dataset (accepted: "baf")
    """
    if not dataset in datasets:
        raise AssertionError

    handler = datasets[dataset]
    X_train, y_train, X_test, y_test, cat_feats = handler.fetch_data()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return {'train':(
      X_train, y_train), 
     'test':(
      X_test, y_test), 
     'cat_feats':cat_feats}


def dump_metrics(metrics, name, metric_file, subtype, model_config, ds_name):

    auc = metrics["AUC"]
    tpr = metrics["TPR"]
    time = metrics["Time"]
    auc_std = 1.96 * metrics["AUC_std"]
    tpr_std = 1.96 * metrics["TPR_std"]
    total_time = time

    print(f"TPR: {tpr} +/- {tpr_std}")
    print(f"AUC: {auc} +/- {auc_std}")


    if not os.path.exists('results/'):
            os.makedirs('results/')
    if not os.path.exists(f'results/{ds_name}'):
            os.makedirs(f'results/{ds_name}/{metric_file}')
    if not os.path.exists(f'results/{ds_name}/{metric_file}'):
            os.makedirs(f'results/{ds_name}/{metric_file}')
    if not os.path.exists(f'results/{ds_name}/{metric_file}/{name}'):
            os.makedirs(f'results/{ds_name}/{metric_file}/{name}')


    # Save the model configuration
    os.makedirs(f'config/', exist_ok=True)
    config_dir = f'config/{ds_name}'
    os.makedirs(config_dir, exist_ok=True)
    with open(config_dir + f"/{name}.pkl", "wb") as fp:
        pickle.dump(model_config, fp)
    
    # Save the model configuration
    os.makedirs(f'info/', exist_ok=True)
    info_dir = f'info/{ds_name}'
    os.makedirs(info_dir, exist_ok=True)
    with open(info_dir + f"/{name}.pkl", "wb") as fp:
        pickle.dump({"time": total_time}, fp)

    if not os.path.isfile(f'results/{ds_name}/{metric_file}/results.csv'):
        df = pd.DataFrame({'type': [subtype], 'name':[name], 
         'TPR':[
          round(tpr, 4)], 
         'TPR_std':[
          round(tpr_std, 4)], 
         'AUC': [
            round(auc, 4)],
         'AUC_std': [
            round(auc_std, 4)],
         'Time':[
          round(time, 2)]})
        df['type'] = pd.Categorical(df['type'], categories=orders[metric_file], ordered=True)
        df.set_index(["type", 'name'], inplace=True, drop=True)
        df = df.sort_values(by = ["type", "name"], ascending = [False, False])
        df.to_csv(f'results/{ds_name}/{metric_file}/results.csv')
    else:
        df = pd.read_csv(f'results/{ds_name}/{metric_file}/results.csv', index_col='name')
        df = df.reset_index()
        df['type'] = pd.Categorical(df['type'], categories=orders[metric_file], ordered=True)
        df.set_index(["type", 'name'], inplace=True, drop=True)
        if (subtype, name) in df.index:
            df.loc[(subtype, name)]['TPR'] = round(tpr, 4)
            df.loc[(subtype, name)]['TPR_std'] = round(tpr_std, 4)
            df.loc[(subtype, name)]['AUC'] = round(auc, 4)
            df.loc[(subtype, name)]['AUC_std'] = round(auc_std, 4)
            df.loc[(subtype, name)]['Time'] = round(time, 4)
        else:
            new_row = pd.DataFrame({'type': [subtype], 'name':[name], 
                'TPR':[
                round(tpr, 4)], 
                'TPR_std':[
                round(tpr_std, 4)], 
                'AUC': [
                    round(auc, 4)],
                'AUC_std': [
                    round(auc_std, 4)],
                'Time':[
                    round(time, 2)]})
            new_row.set_index(['type', 'name'], inplace=True, drop=True)
            df = df.append(new_row)

        df = df.sort_values(by = ["type", "name"], ascending = [True, True])
        df.to_csv(f'results/{ds_name}/{metric_file}/results.csv')
