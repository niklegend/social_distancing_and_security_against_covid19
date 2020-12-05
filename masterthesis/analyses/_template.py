import argparse
import os

import pandas as pd


def analyze(batches_df, epochs_df, hyperparameters_df):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str)

    args = parser.parse_args()
    root = args.root
    del args

    batches_df = pd.read_csv(os.path.join(root, 'batch_logs.csv'))
    epochs_df = pd.read_csv(os.path.join(root, 'epoch_logs.csv'))
    hyperparameters_df = pd.read_csv(os.path.join(root, 'hyperparameters.csv'))

    analyze(batches_df, epochs_df, hyperparameters_df)
