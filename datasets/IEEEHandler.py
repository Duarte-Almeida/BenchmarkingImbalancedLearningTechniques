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

        if not os.path.exists(f"datasets/{self.dir_name}/features.npy"):

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

            print(f"Number of infos: {identity_train_df.shape[0]}")

            merged_df = pd.merge(transaction_train_df, identity_train_df, on='TransactionID', how='inner')

            cat_feats = transaction_cat_feats + identity_cat_feats
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

            # Extract feature importances
            importances = extra_trees.feature_importances_

            # Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]

            #print("Feature ranking:")
            #for f in range(X_train.shape[1]):
            #    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

            # Select features based on importance
            sfm = SelectFromModel(extra_trees, threshold='median', prefit = True)  # You can adjust the threshold if needed

            # Transform the dataset to only include important features
            X_selected = sfm.transform(X)

            # Get a mask of features selected
            selected_features = sfm.get_support()

            #print(f"Selected features: {selected_features}")
            cat_feats = np.array(cat_feats)
            cat_feats = cat_feats[selected_features[-len(cat_feats):]]

            #cat_feats = [c in cat_feats in c in selected_features]

            print(f"Size after reduction: {X_selected.shape[1]}")

            np.save(f"datasets/{self.dir_name}/features.npy", X_selected)
            np.save(f"datasets/{self.dir_name}/targets.npy", y.values)
            with open(f'datasets/{self.dir_name}/info.pkl', 'wb') as fp:
                pkl.dump({"cat_feats": cat_feats}, fp)

            # Plot feature importances in a bar chart
            plt.figure(figsize=(10, 8))
            plt.title("Feature Importances")
            plt.bar(range(X_train.shape[1]), importances[indices], align="center")
            plt.xticks(range(X_train.shape[1]), indices)
            plt.xlim([-1, X_train.shape[1]])
            plt.show()
            plt.clf()
            X = X_selected
            y = y.values
            '''

            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_roc_score:.2f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guessing')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()

            sys.exit()
            '''
        else:
            X = np.load(f"datasets/{self.dir_name}/features.npy")
            y = np.load(f"datasets/{self.dir_name}/targets.npy")
            with open(f"datasets/{self.dir_name}/info.pkl", "rb") as fp:
                cat_feats = pkl.load(fp)["cat_feats"]

            '''
            sample_size = int(0.05 * X.shape[0])
            random_indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sampled = X[random_indices]
            y_sampled = y[random_indices]


            # Assuming the last M columns are categorical variables
            M = len(cat_feats)  # Number of categorical columns (adjust based on your dataset)

            # Step 1: Separate the last M columns and the remaining columns
            continuous_features = X_sampled[:, :-M]
            categorical_features = X_sampled[:, -M:]

            # Step 2: Perform one-hot encoding on the categorical features
            encoder = OneHotEncoder(sparse=False, drop='first')  # Drop the first dummy variable to avoid multicollinearity
            categorical_encoded = encoder.fit_transform(categorical_features)

            # Step 3: Concatenate the one-hot encoded features with the continuous features
            X_processed = np.hstack((continuous_features, categorical_encoded))

            # Step 4: Compute t-SNE representation of X_processed
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X_processed)

            # Step 5: Scatter plot of t-SNE transformed data points colored by target values
            plt.figure(figsize=(10, 8))
            plt.scatter(X_tsne[y_sampled == 0][:, 0], X_tsne[y_sampled == 0][:, 1], color='red', label='Negatives')
            plt.scatter(X_tsne[y_sampled == 1][:, 0], X_tsne[y_sampled == 1][:, 1], color='blue', label='Positives')

            plt.xlabel('t-SNE Feature 1')
            plt.ylabel('t-SNE Feature 2')
            plt.title('t-SNE Visualization with One-Hot Encoded Categorical Features')
            plt.legend()
            plt.grid(True)
            plt.show()
        '''
        
        return X, y, cat_feats