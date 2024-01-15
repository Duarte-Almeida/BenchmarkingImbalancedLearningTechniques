import os, random, numpy as np, pandas as pd, time
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder
import sys
import pickle as pkl
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder

class BAFHandler():

    def __init__(self):
        self.link = 'sgpjesus/bank-account-fraud-dataset-neurips-2022'
        self.dir_name = 'baf'
        self.dataset_file = "Base.csv"
        self.target_name = "fraud_bool"

    def fetch_data(self):

        if not os.path.exists('datasets/'):
            os.makedirs('datasets/')
        if not os.path.exists(f"datasets/{self.dir_name}"):
            os.makedirs(f"datasets/{self.dir_name}")
        if not os.path.exists(f"datasets/{self.dir_name}/{self.dataset_file}"):
            print(f"Dataset {self.dir_name} does not exist. Fetching dataset {self.dir_name}...")
            api = KaggleApi()
            print(f"Fetching {self.dataset_file}")
            api.authenticate()
            api.dataset_download_file(self.link, self.dataset_file, f"datasets/{self.dir_name}")
            zf = ZipFile(f"datasets/{self.dir_name}/{self.dataset_file}.zip")
            zf.extractall(f"datasets/{self.dir_name}")
            zf.close()
            os.remove(f"datasets/{self.dir_name}/{self.dataset_file}.zip")

        if not os.path.exists(f"datasets/{self.dir_name}/features.npy"):

            print(f"Data not processed... Processing data")

            labelencoder = LabelEncoder()
            df = pd.read_csv(f"datasets/{self.dir_name}/{self.dataset_file}")
            
            cat_feats = ["payment_type", "employment_status", "housing_status", 
                         "source","device_os", "month"]
            
            na_feats = ["prev_address_months_count", "current_address_months_count", "bank_months_count", 
                        "session_length_in_minutes"]
            
            non_cat_feats = [col for col in df.columns if col not in cat_feats]
            df = df[non_cat_feats + cat_feats]

            df = df.dropna(axis=1, how='all')
            for col in na_feats:
                df[col].replace(-1, np.nan, inplace = True)
            
            for col in non_cat_feats:
                df[col].fillna(df[col].mean(), inplace=True)
            for col in cat_feats:
                df[col].fillna("Unknown", inplace=True)

            print(f"Encoding categorical variables")

            # Label Encoding
            label_encoders = {}
            for col in cat_feats:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

            X = df.copy()
            X.drop(columns=['fraud_bool'], inplace=True)
            y = df['fraud_bool']

            if X.shape[1] > 50:
                print(f"Performing variable selection based on Extra Trees Classifier")

                # Splitting the data into training and validation sets
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)
                extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)
                extra_trees.fit(X_train, y_train)
                y_valid_probabilities = extra_trees.predict_proba(X_valid)[:, 1]  # Probabilities of the positive class

                # Compute the AUC-ROC score
                auc_roc_score = roc_auc_score(y_valid, y_valid_probabilities)
                fpr, tpr, th = roc_curve(y_valid, y_valid_probabilities)
                fpr_th = 0.05 
                lower_fpr_idx = np.argmax(fpr[fpr < fpr_th])
                lower_th = th[lower_fpr_idx]
                pred_labels = y_valid_probabilities > lower_th

                pos = np.sum(y_valid)
                neg = y_valid.shape[0] - pos

                TP = np.sum(((y_valid == 1) & (pred_labels == 1)))
                FP = np.sum(((y_valid == 0) & (pred_labels == 1)))

                TPR = TP / pos
                FPR = FP / neg

                print(f"AUC: {auc_roc_score}")
                print(f"FPR:{FPR}  \n TPR: {TPR}")
                
                importances = extra_trees.feature_importances_
                indices = np.argsort(importances)[::-1]
                sfm = SelectFromModel(extra_trees, threshold='median', prefit = True)  # You can adjust the threshold if needed

                # Transform the dataset to only include important features
                X_selected = sfm.transform(X)
                selected_features = sfm.get_support()
                cat_feats = np.array(cat_feats)
                cat_feats = cat_feats[selected_features[-len(cat_feats):]]

                print(f"Size after reduction: {X_selected.shape[1]}")

                # Plot feature importances in a bar chart
                plt.figure(figsize=(10, 8))
                plt.title("Feature Importances")
                plt.bar(range(X_train.shape[1]), importances[indices], align="center")
                plt.xticks(range(X_train.shape[1]), indices)
                plt.xlim([-1, X_train.shape[1]])
                plt.show()
                plt.clf()

                X = X_selected

            np.save(f"datasets/{self.dir_name}/features.npy", X.values)
            np.save(f"datasets/{self.dir_name}/targets.npy", y.values)
            with open(f'datasets/{self.dir_name}/info.pkl', 'wb') as fp:
                pkl.dump({"cat_feats": cat_feats}, fp)
            X = X.values
            y = y.values
        
        else:
            X = np.load(f"datasets/{self.dir_name}/features.npy")
            y = np.load(f"datasets/{self.dir_name}/targets.npy")
            with open(f"datasets/{self.dir_name}/info.pkl", "rb") as fp:
                cat_feats = pkl.load(fp)["cat_feats"]
            #with open(f"datasets/{self.dir_name}/res_self.npy", "rb") as fp:
            #    X_ss = np.load(fp)


        return X, y, cat_feats#, X_ss