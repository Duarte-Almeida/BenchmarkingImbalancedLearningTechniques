import os, random, numpy as np, pandas as pd, time
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

datasets = {'baf': ['sgpjesus/bank-account-fraud-dataset-neurips-2022', 'baf', 'Base.csv', 'fraud_bool']}

def fetch_data(dataset):
    """
    Loads the dataset from kaggle, performing a 80-20 train-test split

    dataset: the name of the dataset (accepted: "baf")
    """
    if not dataset in datasets:
        raise AssertionError
    else:
        link = datasets[dataset][0]
        dir_name = datasets[dataset][1]
        dataset_file = datasets[dataset][2]
        target_name = datasets[dataset][3]
        if not os.path.exists('datasets/'):
            os.makedirs('datasets/')
        if not os.path.exists(f"datasets/{dir_name}"):
            os.makedirs(f"datasets/{dir_name}")
        if not os.path.exists(f"datasets/{dir_name}/{dataset_file}"):
            print(f"Dataset {dir_name} does not exist. Fetching dataset {dir_name}...")
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_file(link, dataset_file, f"datasets/{dir_name}")
            zf = ZipFile(f"datasets/{dir_name}/{dataset_file}.zip")
            zf.extractall(f"datasets/{dir_name}")
            zf.close()
            os.remove(f"datasets/{dir_name}/{dataset_file}.zip")
    df = pd.read_csv(f"datasets/{dir_name}/{dataset_file}")
    labelencoder = LabelEncoder()
    cat_feats = df.select_dtypes(include='object').columns
    non_cat_feats = df.select_dtypes(exclude='object').columns
    df = df[list(non_cat_feats) + list(cat_feats)]
    for col in cat_feats:
        df[col] = labelencoder.fit_transform(df[col])

    for col in cat_feats:
        df[col] = df[col].astype('category')

    X = df.drop(columns=target_name)
    y = df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return {'train':(
      X_train, y_train), 
     'test':(
      X_test, y_test), 
     'cat_feats':cat_feats.to_list()}


def dump_metrics(y_test, y_pred, probs, name, time):
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
    G_mean = np.sqrt((1 - FPR) * (1 - FNR))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    auc = roc_auc_score(y_test, probs)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {F1}")
    print(f"False Positive Rate: {FPR}")
    print(f"False Negative Rate: {FNR}")
    print(f"G-mean: {G_mean}")
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    if not os.path.isfile('results.csv'):
        df = pd.DataFrame({'name':[name],  'Precision':[
          round(precision, 4)], 
         'Recall':[
          round(recall, 4)], 
         'F1 score':[
          round(F1, 4)], 
         'False Positive Rate':[
          round(FPR, 4)], 
         'False Negative Rate':[
          round(FNR, 4)], 
         'G-mean':[
          round(G_mean, 4)], 
         'Accuracy':[
          round(accuracy, 4)], 
         'AUC': [
            round(auc, 4)],
         'Time':[
          round(time, 2)]})
        df.set_index('name', inplace=True, drop=True)
        df.to_csv('results.csv')
    else:
        df = pd.read_csv('results.csv', index_col='name')
        if name in df.index:
            df.loc[(name, 'Precision')] = round(precision, 4)
            df.loc[(name, 'Recall')] = round(recall, 4)
            df.loc[(name, 'F1 score')] = round(F1, 4)
            df.loc[(name, 'False Positive Rate')] = round(FPR, 4)
            df.loc[(name, 'False Negative Rate')] = round(FNR, 4)
            df.loc[(name, 'G-mean')] = round(G_mean, 4)
            df.loc[(name, 'Accuracy')] = round(accuracy, 4)
            df.loc[(name, 'AUC')] = round(auc, 4)
            df.loc[(name, 'Time')] = round(time, 2)
        else:
            new_row = pd.DataFrame({'name':[name],  'Precision':[
              round(precision, 4)], 
             'Recall':[
              round(recall, 4)], 
             'F1 score':[
              round(F1, 4)], 
             'False Positive Rate':[
              round(FPR, 4)], 
             'False Negative Rate':[
              round(FNR, 4)], 
             'G-mean':[
              round(G_mean, 4)], 
             'Accuracy':[
              round(accuracy, 4)], 
             'AUC':[
                round(auc, 4)],
             'Time':[
              round(time, 2)]})
            new_row.set_index('name', inplace=True, drop=True)
            df = df.append(new_row)
        df.to_csv('results.csv')
