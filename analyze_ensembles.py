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
from sklearn.metrics import roc_auc_score, roc_curve


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

#ens_names = ["StackedEnsemble", "SelfPaced", "MESA"]
ens_names = ["MESA", "SelfPaced"]
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

for (i, ens) in enumerate(ens_clfs):
    print(f"Training model {ens_names[i]}")
    ens.fit(X_train, y_train)

    # For SelfPaced
    # TODO: Correct for MESA
    predictions = np.zeros((X_test.shape[0], len(ens.estimators_)))

    for j in range(ens.n_estimators):
        predictions[:, j] = ens.estimators_[j].predict_proba(X_test)[:, 1]
        aggregated_preds = np.mean(predictions[:, :j+1], axis=1)
        
        th = thresholds.compute_FPR_cutoff(y_test, aggregated_preds)

        y_pred = (aggregated_preds > th).astype(int)
        aucs[i].append(roc_auc_score(y_test, aggregated_preds))

        TP = np.sum(y_pred[y_test == 1])
        TN = np.sum(1 - y_pred[y_test == 0])
        FP = np.sum(y_pred[y_test == 0])
        FN = np.sum(1 - y_pred[y_test == 1])

        tprs[i].append(TP / (TP + FN))

save_dir = 'analyze/'
os.makedirs(save_dir, exist_ok=True)

for i in range(len(aucs)):
    print(f"Saving {ens_names[i]} plot in {save_dir} ...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.plot(range(1, len(aucs[i])+1), aucs[i], label=f'AUC ({ens_names[i]})', marker='o')
    ax1.set_ylabel('AUC')
    ax1.set_title(f'AUC and TPR vs. Iteration Number for {ens_names[i]}')
    ax1.grid(True)

    ax2.plot(range(1, len(tprs[i])+1), tprs[i], label=f'TPR ({ens_names[i]})', marker='o')
    ax2.set_xlabel('Iteration Number')
    ax2.set_ylabel('TPR')
    ax2.grid(True)

    plt.tight_layout()

    save_path = os.path.join(save_dir, f'Ensemble_{ens_names[i]}_plot.png')
    plt.savefig(save_path)
    plt.clf()

    print(f"Saved {ens_names[i]} plot in {save_path} !")