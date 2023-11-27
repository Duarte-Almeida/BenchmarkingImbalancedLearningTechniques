import argparse
import random
import os
import time

import utils

import lightgbm as lgb
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_metrics(y_test, y_pred):
    TP = np.sum(y_pred[y_test == 1])
    TN = np.sum(1 - y_pred[y_test == 0])
    FP = np.sum(y_pred[y_test == 0])
    FN = np.sum(1 - y_pred[y_test == 1])

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    F1 = 2 * (precision * recall) / (precision + recall)

    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)

    G_mean = np.sqrt((1 - FPR) * (1 - FNR))

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {F1}")
    print(f"False Positive Rate: {FPR}")
    print(f"False Negative Rate: {FNR}")
    print(f"G-mean: {G_mean}")
    print(f"Accuracy: {accuracy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset',
                        choices=['baf'],
                        help="Which dataset should we choose?")
    parser.add_argument('-oversampling', default = None,
                        choices=['TODO'],
                        help="Which oversampling strategy should we choose?")
    parser.add_argument('-undersampling', default = None,
                        choices=['TODO'],
                        help="Which undersampling strategy should we choose?")
    parser.add_argument('-hybrid_over_under', default = None,
                        choices=['TODO'],
                        help="Which hybrid oversampling/undersammpling strategy should we choose?")
    opt = parser.parse_args()

    data = utils.fetch_data(opt.dataset)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    cat_feats = data["cat_feats"]

    start_time = time.time()

    print('Starting training...')
    clf = lgb.LGBMClassifier()
    clf.fit(X = X_train, y = y_train, categorical_feature = cat_feats)
    y_pred = clf.predict(X_test)

    end_time = time.time()
    print(f'Ended training and classification in {round(end_time - start_time, 2)} seconds.\n')

    compute_metrics(y_test, y_pred)


if __name__ == '__main__':
    main()