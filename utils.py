import os, random, numpy as np, pandas as pd, time
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import pickle
import sys

datasets = {'baf': ['sgpjesus/bank-account-fraud-dataset-neurips-2022', 'baf', 'Base.csv', 'fraud_bool']}

orders = {
    "preprocessing": ["Base", "Oversamplers", "Undersamplers", "Hybrid", "Label Smoothing"],
    "inprocessing": ["FocalLoss", "GradientHarmonized"]
}

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


def dump_metrics(y_test, y_pred, probs, name, total_time, test_time, metric_file, subtype, model_config):
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

    '''
    # compute total expected calibration error
    M = 100
    eps = 1 / M
    C_histogram = np.zeros(M)
    A_histogram = np.zeros(M)
    counts = np.zeros(M)

    C_pos_histogram = np.zeros(M)
    A_pos_histogram = np.zeros(M)
    counts_pos = np.zeros(M)

    C_neg_histogram = np.zeros(M)
    A_neg_histogram = np.zeros(M)
    counts_neg = np.zeros(M)
   
    for (prob, true, pred) in zip(probs, y_test, y_pred):
        conf = prob * pred + (1 - prob) * (1 - pred)
        idx = min(int(np.floor(conf / eps)), M - 1)
        counts[idx] += 1
        C_histogram[idx] += conf
        A_histogram[idx] += (pred == true)
        if true:
            counts_pos[idx] += 1
            C_pos_histogram[idx] += conf
            A_pos_histogram[idx] += (pred == true)
        else:
            counts_neg[idx] += 1
            C_neg_histogram[idx] += conf
            A_neg_histogram[idx] += (pred == true)

    N = y_test.shape[0]
    N_pos = y_test[y_test == 1].shape[0]
    N_neg = y_test[y_test == 0].shape[0]
    print(f"Total Calibration Error: {np.sum(np.abs(C_histogram - A_histogram)) / N}")
    print(f"Positive Calibration Error: {np.sum(np.abs(C_pos_histogram - A_pos_histogram)) / N}")
    print(f"Negative Calibration Error: {np.sum(np.abs(C_neg_histogram - A_neg_histogram)) / N}")

    C_histogram[C_histogram != 0] = (C_histogram[C_histogram != 0] / counts[C_histogram != 0]) #/ counts[C_histogram != 0]) * N
    A_histogram[A_histogram != 0] = (A_histogram[A_histogram != 0] / counts[A_histogram != 0]) #/ counts[A_histogram != 0]) * N
    histogram = C_histogram - A_histogram
    plt.bar([i for i in range(1, M + 1)], histogram)
    plt.show()
    plt.clf()

    C_pos_histogram[C_pos_histogram != 0] = (C_pos_histogram[C_pos_histogram != 0] / counts[C_pos_histogram != 0])
    A_pos_histogram[A_pos_histogram != 0] = (A_pos_histogram[A_pos_histogram != 0] / counts[A_pos_histogram != 0])
    histogram_pos = C_pos_histogram - A_pos_histogram
    plt.bar([i for i in range(1, M + 1)], histogram_pos, label = "Positive", alpha = 0.5)
    C_neg_histogram[C_neg_histogram != 0] = (C_neg_histogram[C_neg_histogram != 0] / counts[C_neg_histogram != 0]) 
    A_neg_histogram[A_neg_histogram != 0] = (A_neg_histogram[A_neg_histogram != 0] / counts[A_neg_histogram != 0])
    histogram_neg = C_neg_histogram - A_neg_histogram
    plt.bar([i for i in range(1, M + 1)], histogram_neg, label = "Negative", alpha = 0.5)
    plt.legend()
    plt.show()
    plt.clf()
    '''
    if not os.path.exists('results/'):
            os.makedirs('results/')
    if not os.path.exists(f'results/{metric_file}'):
            os.makedirs(f'results/{metric_file}')
    if not os.path.exists(f'results/{metric_file}/{name}'):
            os.makedirs(f'results/{metric_file}/{name}')

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
    config_dir = f'config/'
    os.makedirs(config_dir, exist_ok=True)
    with open(config_dir + f"/{name}.pkl", "wb") as fp:
        pickle.dump(model_config, fp)
    
    # Save the model configuration
    info_dir = f'info/'
    os.makedirs(info_dir, exist_ok=True)
    with open(info_dir + f"/{name}.pkl", "wb") as fp:
        pickle.dump({"time": total_time}, fp)

    # save the figure
    results_dir = f'results/{metric_file}/{name}'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'{name}_roc.pdf'))
    plt.clf()
    if not os.path.isfile(f'results/{metric_file}/results.csv'):
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
        df.to_csv(f'results/{metric_file}/results.csv')
    else:
        df = pd.read_csv(f'results/{metric_file}/results.csv', index_col='name')
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
        df.to_csv(f'results/{metric_file}/results.csv')
