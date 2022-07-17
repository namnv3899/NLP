import logging
import os
import time

import yaml
import pandas as pd
from sklearn.utils import shuffle

LOG_DIR = os.path.join("logs")
LOG_FORMAT = "%(levelname)s %(name)s %(asctime)s - %(message)s"

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

log_filename = os.path.join(LOG_DIR, "crawling.log")


def split_data(df, train_size, is_shuffle=True):
  train_dfs = []
  test_dfs = []
  categories = df.category.unique()

  for cat in categories:
    cat_data = df[df.category == cat]
    if is_shuffle:
        cat_data = shuffle(cat_data)
    cat_data.reset_index(inplace=True, drop=True)

    n_train_rows = int(cat_data.shape[0] * train_size) if train_size < 1 else train_size

    train_dfs.append(cat_data.iloc[:n_train_rows, :])
    test_dfs.append(cat_data.iloc[n_train_rows:, :])

    train_df = pd.concat(train_dfs, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_dfs, axis=0).reset_index(drop=True)

    return train_df, test_df

def get_logger(logger_name):
    logging.basicConfig(filename=log_filename, level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger(logger_name)

    return logger


def load_yaml(field, path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)
    return config[field]


def load_stopwords():
    path = load_yaml("paths")["stopwords"]
    with open(path, "r") as f:
        stopwords = f.read().split("\n")
    return stopwords


def timing(method):
    def timed(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()

        execution_time = end - start
        if execution_time < 0.001:
            print(
                f"{method.__name__} took {round(execution_time * 1000, 3)} milliseconds"
            )
        else:
            print(f"{method.__name__} took {round(execution_time, 3)} seconds")

        return result

    return timed
