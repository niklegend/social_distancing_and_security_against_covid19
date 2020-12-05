import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_train_val(ax, metric, train, val, num_epochs, loc):
    ax.plot(np.arange(1, num_epochs + 1), train.values, label=f'Train {metric}')
    ax.plot(np.arange(1, num_epochs + 1), val.values, label=f'Validation {metric}')

    ax.set_xlim(left=1)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.capitalize())

    ax.set_title(f'Train/val {metric}')

    ax.legend(loc=loc)


def plot_batch_epoch(ax, phase, metric, batch, epoch, num_epochs, num_batches_per_epoch, loc):
    ax.plot(batch.values, label=f'Batch {metric}')
    ax.plot(np.arange(1, num_epochs + 1) * num_batches_per_epoch, epoch.values, label=f'Epoch {metric}')

    ax.set_xlim(left=1)

    ax.set_xlabel('Batch/Epoch')
    ax.set_ylabel(metric.capitalize())

    ax.set_title(f'Batch/Epoch {phase} {metric}')

    ax.legend(loc=loc)


def analyze(batches_df, epochs_df, hyperparameters_df):
    train_batches = batches_df[batches_df['phase'] == 'train']
    val_batches = batches_df[batches_df['phase'] == 'val']

    num_epochs = len(epochs_df)
    train_num_batches = epochs_df.iloc[0]['train_num_batches']
    val_num_batches = epochs_df.iloc[0]['val_num_batches']
    print(f'Num epochs: {num_epochs}')
    print(f'Num batches: {train_num_batches}')

    fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3)

    plot_train_val(ax11, 'loss', epochs_df['train_loss'], epochs_df['val_loss'], num_epochs, 'upper right')
    plot_train_val(ax21, 'accuracy', epochs_df['train_acc'], epochs_df['val_acc'], num_epochs, 'lower right')

    plot_batch_epoch(ax12, 'train', 'loss', train_batches['loss'], epochs_df['train_loss'], num_epochs,
                     train_num_batches,
                     'upper right')
    plot_batch_epoch(ax13, 'val', 'loss', val_batches['loss'], epochs_df['val_loss'], num_epochs, val_num_batches,
                     'upper right')

    plot_batch_epoch(ax22, 'train', 'acc', train_batches['acc'], epochs_df['train_acc'], num_epochs, train_num_batches,
                     'lower right')
    plot_batch_epoch(ax23, 'val', 'acc', val_batches['acc'], epochs_df['val_acc'], num_epochs, val_num_batches,
                     'lower right')

    plt.subplots_adjust(hspace=0.5)

    fig.suptitle(f'{hyperparameters_df["Model"]} epoch statistics')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str)

    args = parser.parse_args()
    root = args.root
    del args

    b = pd.read_csv(os.path.join(root, 'batches.csv'))
    e = pd.read_csv(os.path.join(root, 'epochs.csv'))
    h = pd.read_csv(os.path.join(root, 'hyperparameters.csv'))

    analyze(b, e, h.iloc[0])
