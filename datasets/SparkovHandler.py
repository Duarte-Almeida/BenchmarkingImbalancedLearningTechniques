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

class SparkovHandler():

    def __init__(self):
        self.link = 'kartik2112/fraud-detection'
        self.dir_name = 'sparkov'
        self.dataset_file = "fraudTrain.csv"
        self.target_name = "is_fraud"

    def fetch_data(self):
        
        if not os.path.exists(f"datasets/{self.dir_name}/features_train.npy"):
            if not os.path.exists('datasets/'):
                os.makedirs('datasets/')
            if not os.path.exists(f"datasets/{self.dir_name}"):
                os.makedirs(f"datasets/{self.dir_name}")

            file = self.dataset_file
            if not os.path.exists(f"datasets/{self.dir_name}/{file}"):
                print(f"Dataset {self.dir_name} does not exist. Fetching dataset {file}...")
                api = KaggleApi()
                print(f"Fetching {file}")
                api.authenticate()
                api.dataset_download_file(self.link, file, f"datasets/{self.dir_name}")
                zf = ZipFile(f"datasets/{self.dir_name}/{file}.zip")
                zf.extractall(f"datasets/{self.dir_name}")
                zf.close()
                os.remove(f"datasets/{self.dir_name}/{file}.zip")

            print(f"Data not processed... Processing data")
            df = pd.read_csv(f"datasets/{self.dir_name}/{file}")
            
            cat_feats = ["merchant", "category", "gender", "street",
                        "city", "state", "job"]
            
            df = df.drop(columns=df.columns[0])
            df = df.drop(columns=["trans_date_trans_time", "cc_num", "first", "last", "trans_num"])
            df["dob"] = pd.to_datetime(df["dob"]).astype(int) / (10 ** 9)

            # drop attributes that uniquely identify a transaction
            non_cat_feats = [col for col in df.columns if col not in cat_feats]
            df = df[non_cat_feats + cat_feats]

            df = df.dropna(axis=1, how='all')
            
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
            X.drop(columns=[self.target_name], inplace=True)
            y = df[self.target_name]

            if X.shape[1] > 50:
                print(f"Performing variable selection based on Extra Trees Classifier")

                # Splitting the data into training and validation sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)
                extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)
                extra_trees.fit(X_train, y_train)

                '''
                y_test_probabilities = extra_trees.predict_proba(X_test)[:, 1]  # Probabilities of the positive class

                # Compute the AUC-ROC score
                auc_roc_score = roc_auc_score(y_test, y_test_probabilities)
                fpr, tpr, th = roc_curve(y_test, y_test_probabilities)
                fpr_th = 0.05 
                lower_fpr_idx = np.argmax(fpr[fpr < fpr_th])
                lower_th = th[lower_fpr_idx]
                pred_labels = y_test_probabilities > lower_th

                pos = np.sum(y_test)
                neg = y_test.shape[0] - pos

                TP = np.sum(((y_test == 1) & (pred_labels == 1)))
                FP = np.sum(((y_test == 0) & (pred_labels == 1)))

                TPR = TP / pos
                FPR = FP / neg

                print(f"AUC: {auc_roc_score}")
                print(f"FPR:{FPR}  \n TPR: {TPR}")
                '''
                
                importances = extra_trees.feature_importances_
                indices = np.argsort(importances)[::-1]
                sfm = SelectFromModel(extra_trees, threshold='median', prefit = True)  

                # Transform the dataset to only include important features
                X_selected_train = sfm.transform(X_train)
                X_selected_test = sfm.transform(X_test)
                selected_features = sfm.get_support()
                cat_feats = np.array(cat_feats)
                cat_feats = cat_feats[selected_features[-len(cat_feats):]]

                X_train = X_selected_train
                X_test = X_selected_test
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)

    
            np.save(f"datasets/{self.dir_name}/features_train.npy", X_train.values)
            np.save(f"datasets/{self.dir_name}/targets_train.npy", y_train.values)
            np.save(f"datasets/{self.dir_name}/features_test.npy", X_test.values)
            np.save(f"datasets/{self.dir_name}/targets_test.npy", y_test.values)
            with open(f'datasets/{self.dir_name}/info.pkl', 'wb') as fp:
                pkl.dump({"cat_feats": cat_feats}, fp)
            X_train = X_train.values
            y_train = y_train.values
            X_test = X_test.values
            y_test = y_test.values
        
        else:
            X_train = np.load(f"datasets/{self.dir_name}/features_train.npy")
            y_train = np.load(f"datasets/{self.dir_name}/targets_train.npy")
            X_test = np.load(f"datasets/{self.dir_name}/features_test.npy")
            y_test = np.load(f"datasets/{self.dir_name}/targets_test.npy")
            print(f"Size: {X_train.shape}")
            print(f"Size: {y_train.shape}")

            print(f"Size: {X_test.shape}")
            print(f"Size: {y_test.shape}")
            with open(f"datasets/{self.dir_name}/info.pkl", "rb") as fp:
                cat_feats = pkl.load(fp)["cat_feats"]

        return X_train, y_train, X_test, y_test, cat_feats