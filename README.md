# Benchmarking robust model training techniques in the imbalanced setting

## Setup

### Generate Kaggle API Token

To fetch datasets from Kaggle, we have to generate an API token into a json file. For that purpose, in the Kaggle site, go the API section in the Settings page, click "Create new token" and save the generated ```kaggle.json``` file in this directory. Then run the following commands:

```sh
sudo mkdir .kaggle
sudo cp kaggle.json .kaggle
sudo chmod 600 .kaggle/kaggle.json
sudo chown `whoami`: .kaggle/kaggle.json
export KAGGLE_CONFIG_DIR='.kaggle/'
```

### Dependencies

To install the dependencies, run:

```sh
pip install Cython==0.29.24
pip install numpy==1.21.6
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -r requirements.txt
```

## Running the project

To get metrics from applying this baseline classifier, run the following command:

```sh
python3 main.py -dataset <dataset> [-data_subsampling <proportion>]
[-oversampling <oversampling_strategy>] [-undersampling <oversampling_strategy>]
[-clf <classifier>] [-loss <loss_function>] [-plot_scores] [-simplified]
[-ensemble <ensemble_method>] [-n_iter <value>]
```

The possible choices are:
- `-dataset`: `baf` | `mlg`;
- `-data_subsampling`: 0 < `proportion` < 1;
- `-oversampling`: `SMOTE` | `ADASYN` | `KMeansSMOTE` | `RACOG`;
- `-undersampling`: `RUS` | `NCR` | `IHT`;
- `-clf`: `Base`;
- `-loss`: `WeightedCrossEntropy` | `LabelSmoothing` | `LabelRelaxation` | `FocalLoss` | `GradientHarmonized`;
- `-ensemble`: `StackedEnsemble` | `SelfPaced`;
- `-n_iter`: positive integer value (default = 50);

We recommend using Python 3.7.

A `dataset` must be chosen manually. [BAF](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data) and [MLG](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) are two datasets from Kaggle. Only on the first time will the dataset be fetched from Kaggle. In subsequent executions, it will be stored in memory.

`data_subsampling` determines the amount of training examples to retain from the original training set (useful for very big datasets)

`oversampling` / `undersampling` / `ensemble` performs the chosen oversampling / undersampling / ensemble method. In addition, these methods also support categorical data.

The base classifier is an adaptation of [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html).

`loss` changes the base classifier's loss function to the specified one.

`n_iter` states how many iterations of the hyperparameter random search should be performed. This search is implemented with [TPE](https://arxiv.org/abs/2304.11127).

`plot_scores` allows for a visualization of the class conditional distribution of the model's scores.




