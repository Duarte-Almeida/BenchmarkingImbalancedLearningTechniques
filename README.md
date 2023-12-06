# Benchmarking robust model training techniques in the imbalanced setting

## Setup

### Generate Kaggle API Token

To fetch datasets from kaggle, we have to generate an API token into a json file. For that purpose, in the Kaggle site, go the API section in the Settings page, click "Create new token" and save the generated ```kaggle.json``` file in this directory. Then run the following commands:

```sh
sudo mkdir .kaggle
sudo cp kaggle.json .kaggle
sudo chmod 600 .kaggle/kaggle.json
sudo chown `whoami`: .kaggle/kaggle.json
export KAGGLE_CONFIG_DIR='.kaggle/'
```

### Dependencies

To install the dependencies, just run:

```sh
pip install -r requirements.txt
```

## Running the project

To get metrics from applying this baseline classifier, run the following command:

```sh
python3 main.py -dataset baf [-oversampling <oversampling_strategy>] [-undersampling <oversampling_strategy>] [-label_smoothing] [-plot_scores]
```

If -label_smoothing is set, the targets are smoothed and -plot_scores allows for a visualization of the class conditional distribution of the model's scores. As of now, the following undersamping and oversampling strategies are supported
- <oversampling_strategy> can be SMOTE or ADASYN
- <undersampling_strategy> can be Random Undersampling

Only on the first time will the dataset be fetched from Kaggle. In subsequent executions, it will be stored in memory.




