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

As of now, this project only contains a a LightGBM classifier without any extra techniques and only supports the BAF dataset. To get metrics from applying this baseline classifier, run the following command:

```sh
python3 main.py -dataset baf
```

Only on the first time will the dataset be fetched from Kaggle. In subsequent executions, it will be stored in memory.




