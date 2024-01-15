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
    X, y, cat_feats = handler.fetch_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return {'train':(
      X_train, y_train), 
     'test':(
      X_test, y_test), 
     'cat_feats':cat_feats}


def dump_metrics(y_test, y_pred, probs, name, total_time, test_time, metric_file, subtype, model_config, ds_name):
    TP = np.sum(y_pred[y_test == 1])
    TN = np.sum(1 - y_pred[y_test == 0])
    FP = np.sum(y_pred[y_test == 0])
    FN = np.sum(1 - y_pred[y_test == 1])
    print(f"True positives: {TP}")
    print(f"False positives: {FP}")
    print(f"True negatives: {TN}")
    print(f"False negatives: {FN}")
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision == 0 or recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)
    if FP + TN == 0:
        FPR = 0
    else:
        FPR = FP / (FP + TN)
    if FN + TP == 0:
        FNR = 0
    else:
        FNR = FN / (FN + TP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {F1}")
    print(f"False Positive Rate: {FPR}")
    print(f"False Negative Rate: {FNR}")
    print(f"AUC: {auc}")
    #sys.exit()
    if not os.path.exists('results/'):
            os.makedirs('results/')
    if not os.path.exists(f'results/{ds_name}'):
            os.makedirs(f'results/{ds_name}/{metric_file}')
    if not os.path.exists(f'results/{ds_name}/{metric_file}'):
            os.makedirs(f'results/{ds_name}/{metric_file}')
    if not os.path.exists(f'results/{ds_name}/{metric_file}/{name}'):
            os.makedirs(f'results/{ds_name}/{metric_file}/{name}')

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name.replace('_', ' '))
    plt.legend(loc='lower right')

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

    # save the figure
    results_dir = f'results/{ds_name}/{metric_file}/{name}'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'{name}_roc.pdf'))
    plt.clf()
    if not os.path.isfile(f'results/{ds_name}/{metric_file}/results.csv'):
        df = pd.DataFrame({'type': [subtype], 'name':[name], 'Precision':[
          round(precision, 4)], 
         'Recall':[
          round(recall, 4)], 
         'F1 score':[
          round(F1, 4)], 
         'False Positive Rate':[
          round(FPR, 4)], 
         'False Negative Rate':[
          round(FNR, 4)], 
         'AUC': [
            round(auc, 4)],
         'Train time':[
          round(total_time, 2)],
         'Test time':[
          round(test_time, 2)]})
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
            df.loc[(subtype, name)]['Precision'] = round(precision, 4)
            df.loc[(subtype, name)]['Recall'] = round(recall, 4)
            df.loc[(subtype, name)]['F1 score'] = round(F1, 4)
            df.loc[(subtype, name)]['False Positive Rate'] = round(FPR, 4)
            df.loc[(subtype, name)]['False Negative Rate'] = round(FNR, 4)
            df.loc[(subtype, name)]['AUC'] = round(auc, 4)
            df.loc[(subtype, name)]['Train time'] = round(total_time, 2)
            df.loc[(subtype, name)]['Test time'] = round(test_time, 2)
        else:
            new_row = pd.DataFrame({'type': [subtype], 'name':[name],  'Precision':[
              round(precision, 4)], 
             'Recall':[
              round(recall, 4)], 
             'F1 score':[
              round(F1, 4)], 
             'False Positive Rate':[
              round(FPR, 4)], 
             'False Negative Rate':[
              round(FNR, 4)], 
             'AUC':[
                round(auc, 4)],
             'Train time':[
              round(total_time, 2)],
              'Test time':[
              round(test_time, 2)]})
            new_row.set_index(['type', 'name'], inplace=True, drop=True)
            df = df.append(new_row)

        df = df.sort_values(by = ["type", "name"], ascending = [True, True])
        df.to_csv(f'results/{ds_name}/{metric_file}/results.csv')
