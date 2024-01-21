import os
import utils
import pickle as pkl
from functools import *
from classifiers import LGBM
import ensembles
import thresholds
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

def estimator_prediction(estimator, predictions, X_test, aggregate):
    predictions[:, j] = estimator.predict_proba(X_test)[:, 1]
    if aggregate:
        aggregated_preds = np.mean(predictions[:, :j+1], axis=1)
    else:
        aggregated_preds = predictions[:, j]
    
    th = thresholds.compute_FPR_cutoff(y_test, aggregated_preds)

    y_pred = (aggregated_preds > th).astype(int)

    return y_pred, aggregated_preds


np.random.seed(42)
dataset = "baf"

data = utils.fetch_data(dataset)
X_train, y_train = data["train"]
X_test, y_test = data["test"]
cat_feats = data["cat_feats"]

sub_idx = np.random.choice(X_train.shape[0], size = int(X_train.shape[0] * 0.1), replace = False)
X_train = X_train[sub_idx]
y_train = y_train[sub_idx]

base_models = [LGBM() for _ in range(3)]
for (i, base_model) in enumerate(base_models):
    base_model.set_params(**{"categorical_features":len(cat_feats)})

# SelfPaced and MESA must be sequentially in the list!
ens_names = ["StackedEnsemble", "SelfPaced", "MESA"]
ens_clfs = []
for i in range(len(ens_names)):
    ens_clfs.append(ensembles.fetch_ensemble(ens_names[i]))

for (ens, ens_name, base_model) in zip(ens_clfs, ens_names, base_models):
    ens.set_params(**{"base_estimator": base_model})
    with open(f"config/{dataset}/{ens_name}.pkl", "rb") as fp:
        ens_config = pkl.load(fp)
    for (param, value) in ens_config.items():
        ens.set_params(**{param[len("ens__"):]:value})

aucs = [[] for _ in range(len(ens_names))]
tprs = [[] for _ in range(len(ens_names))]
y_preds = [[] for _ in range(len(ens_names))]

for (i, ens) in enumerate(ens_clfs):
    print(f"Training model {ens_names[i]}")
    ens.fit(X_train, y_train)

    # For SelfPaced & MESA (mesa also has the meta training)
    predictions = np.zeros((X_test.shape[0], len(ens.estimators_)))

    for j in range(ens.n_estimators):
        y_pred, aggregated_preds = estimator_prediction(
            ens.estimators_[j], predictions, X_test, ens_names[i] != "StackedEnsemble")

        y_preds[i].append(aggregated_preds)
        aucs[i].append(roc_auc_score(y_test, aggregated_preds))

        TP = np.sum(y_pred[y_test == 1])
        TN = np.sum(1 - y_pred[y_test == 0])
        FP = np.sum(y_pred[y_test == 0])
        FN = np.sum(1 - y_pred[y_test == 1])

        tprs[i].append(TP / (TP + FN))
    
    if ens_names[i] == "StackedEnsemble":
        meta_X = np.zeros((X_test.shape[0], ens.n_estimators))
        for j in range(ens.n_estimators):
            meta_X[:, j] = y_preds[i][j]

        _, aggregated_preds = estimator_prediction(
            ens.meta_learner, predictions, meta_X, False)
        y_preds[i].append(aggregated_preds)

save_dir = 'analysis/'
os.makedirs(save_dir, exist_ok=True)
first = True

for i in range(len(aucs)):
    print(f"Saving {ens_names[i]} plot in {save_dir} ...")

    if ens_names[i] == "StackedEnsemble":
        plt.figure(figsize=(10, 6))
        for j, classifier_preds in enumerate(y_preds[i][:-1]):
            fpr, tpr, _ = roc_curve(y_test, classifier_preds)
            roc_auc = auc(fpr, tpr)
            print(f'AUC for Classifier {j+1}: {roc_auc}')
            plt.plot(fpr, tpr, label=f'Classifier {j+1} (AUC = {roc_auc:.2f})')

        fpr_base, tpr_base, _ = roc_curve(y_test, y_preds[i][-1])
        roc_auc_base = auc(fpr_base, tpr_base)
        print(f'AUC for Meta Learner: {roc_auc_base}')
        plt.plot(fpr_base, tpr_base, label=f'Baseline Classifier (AUC = {roc_auc_base:.2f})', linestyle='--')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()

        save_path = os.path.join(save_dir, f'Ensemble_{ens_names[i]}_{dataset}.png')
        plt.savefig(save_path)
        plt.clf()

        print(f"Saved {ens_names[i]} plot in {save_path} !")
        
    else :
        if first:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

        ax1.plot(range(1, len(aucs[i])+1), aucs[i], label=f'AUC ({ens_names[i]})', marker='o')
        ax1.set_ylabel('AUC')
        ax1.set_title(f'AUC and TPR vs. Iteration Number for {ens_names[i]}')
        ax1.grid(True)

        ax2.plot(range(1, len(tprs[i])+1), tprs[i], label=f'TPR ({ens_names[i]})', marker='o')
        ax2.set_xlabel('Iteration Number')
        ax2.set_ylabel('TPR')
        ax2.grid(True)

        first = False

plt.legend()
plt.tight_layout()

save_path = os.path.join(save_dir, f'Ensemble_{ens_names[i-1]}&{ens_names[i]}_{dataset}.png')
plt.savefig(save_path)
plt.clf()

print(f"Saved {ens_names[i]} plot in {save_path} !")