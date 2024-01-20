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

class IEEEHandler():

    def __init__(self):
        self.link = 'ieee-fraud-detection'
        self.dir_name = 'ieee'
        self.transactions_train = "train_transaction.csv"
        self.transactions_test = "test_transaction.csv"
        self.identities_train = "train_identity.csv"
        self.identities_test = "test_identity.csv"
        self.dataset_files = [
            self.transactions_train,
            self.identities_train,
        ]

    def fetch_data(self):
        
        if not os.path.exists(f"datasets/{self.dir_name}/features_train.npy"):
            if not os.path.exists('datasets/'):
                os.makedirs('datasets/')
            if not os.path.exists(f"datasets/{self.dir_name}"):
                os.makedirs(f"datasets/{self.dir_name}")
            for file in self.dataset_files:
                if os.path.exists(f"datasets/{self.dir_name}/{file}"):
                    continue
                print(f"Dataset {self.dir_name} does not exist. Fetching dataset {self.dir_name}...")
                api = KaggleApi()
                print(f"Fetching {file}")
                api.authenticate()
                api.competition_download_file(self.link, file, f"datasets/{self.dir_name}")
                zf = ZipFile(f"datasets/{self.dir_name}/{file}.zip")
                zf.extractall(f"datasets/{self.dir_name}")
                zf.close()
                os.remove(f"datasets/{self.dir_name}/{file}.zip")

            print(f"Data not processed... Processing data")

            labelencoder = LabelEncoder()
            
            # Process the transaction part
            transaction_cat_feats = ["ProductCD"] + [f"card{i}" for i in range(1, 7)] + \
                                    [f"addr{i}" for i in range(1, 3)] + \
                                    ["P_emaildomain", "R_emaildomain"] + \
                                    [f"M{i}" for i in range(1, 10)]
            transaction_train_df = pd.read_csv(f"datasets/{self.dir_name}/{self.transactions_train}")

            print(f"Number of transactions: {transaction_train_df.shape[0]}")

            identity_cat_feats = ["DeviceType", "DeviceInfo"] + [f"id_{i}" for i in range(12, 39)]
            identity_train_df = pd.read_csv(f"datasets/{self.dir_name}/{self.identities_train}")

            #print(f"Number of infos: {identity_train_df.shape[0]}")

            #merged_df = pd.merge(transaction_train_df, identity_train_df, on='TransactionID', how='inner')
            merged_df = transaction_train_df

            cat_feats = transaction_cat_feats #+ identity_cat_feats
            non_cat_feats = [col for col in merged_df.columns if col not in cat_feats]

            columns_with_nan = merged_df.columns[merged_df.isna().any()].tolist()

            for col in non_cat_feats:
                merged_df[col].fillna(merged_df[col].mean(), inplace=True)
            for col in cat_feats:
                merged_df[col].fillna("Unknown", inplace=True)
            
            print(f"Encoding categorical variables")

            # Label Encoding
            label_encoders = {}
            for col in cat_feats:
                le = LabelEncoder()
                merged_df[col] = le.fit_transform(merged_df[col].astype(str))
                label_encoders[col] = le

            # Reorganize Data
            X = merged_df[non_cat_feats + cat_feats]
            X.drop(columns=['TransactionID', 'TransactionDT', 'isFraud'], inplace=True)
            X = X.dropna(axis=1, how='all')
            X = X

            y = merged_df['isFraud']

            print(f"Performing variable selection based on Extra Trees Classifier")

            # Splitting the data into training and validation sets
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=42, shuffle = True)
            extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)
            extra_trees.fit(X_train, y_train)

            '''
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
            '''
            # Select the top 60 features
            sfm = SelectFromModel(extra_trees, prefit=True, max_features=30)

            # Get the indices of the selected features
            print(np.where(sfm.get_support()))
            selected_feature_indices = np.where(sfm.get_support())[0]

            # Retain the order of the features in the original data
            X_selected_train = X_train.values[:, selected_feature_indices]
            X_selected_test = X_valid.values[:, selected_feature_indices]

            selected_features = sfm.get_support()
            print(f"Here are my selected features: {selected_features}")
            print(f"Here are my selected cat features: {selected_features[-len(cat_feats):]}")
            cat_feats = np.array(cat_feats)
            cat_feats = cat_feats[selected_features[-len(cat_feats):]]

            X_train = X_selected_train
            X_test = X_selected_test
            y_test = y_valid

            np.save(f"datasets/{self.dir_name}/features_train.npy", X_train)
            np.save(f"datasets/{self.dir_name}/targets_train.npy", y_train.values)
            np.save(f"datasets/{self.dir_name}/features_test.npy", X_test)
            np.save(f"datasets/{self.dir_name}/targets_test.npy", y_test.values)
            with open(f'datasets/{self.dir_name}/info.pkl', 'wb') as fp:
                pkl.dump({"cat_feats": cat_feats}, fp)
            X_train = X_train
            y_train = y_train.values
            X_test = X_test
            y_test = y_test.values
        
        else:
            X_train = np.load(f"datasets/{self.dir_name}/features_train.npy")
            y_train = np.load(f"datasets/{self.dir_name}/targets_train.npy")
            X_test = np.load(f"datasets/{self.dir_name}/features_test.npy")
            y_test = np.load(f"datasets/{self.dir_name}/targets_test.npy")
            with open(f"datasets/{self.dir_name}/info.pkl", "rb") as fp:
                cat_feats = pkl.load(fp)["cat_feats"]

        return X_train, y_train, X_test, y_test, cat_feats