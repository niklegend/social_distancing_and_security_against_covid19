import multiprocessing
import os

import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from masterthesis.experiment import Engine, metrics
from masterthesis.experiment.callbacks import CsvCallback


class Flatten(object):

    def __call__(self, x):
        return x.reshape(-1)


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MNISTEngine(Engine):
    def __init__(self, module, optimizer, criterion):
        super(MNISTEngine, self).__init__(module, optimizer)
        self.criterion = criterion

    def compute_loss(self, is_train, outputs, targets):
        return self.criterion(outputs, targets), outputs


# Model hyper-parameters
input_size = 784
hidden_size = 40
num_classes = 10

# DataLoaders hyper-parameters
batch_size = 64
num_workers = multiprocessing.cpu_count()
shuffle = True

transform = Compose([
    ToTensor(),
    Flatten()
])

# Optimizer hyper-parameters
learning_rate = 1e-3

# Training hyper-parameters
num_epochs = 1000
progressbar = True

root = 'data'
filepath = os.path.join(root, 'checkpoint.pth')
save = False


def create_loader(dataset, sampler=None, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, shuffle=shuffle)


if __name__ == '__main__':
    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.1)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = create_loader(train_dataset, sampler=train_sampler)
    val_loader = create_loader(train_dataset, sampler=val_sampler)

    model = MLP(input_size, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    engine = MNISTEngine(model, optimizer, criterion)

    engine.fit(
        train_loader, num_epochs,
        val_loader=val_loader,
        progressbar=progressbar,
        metrics=[metrics.Accuracy()],
        callbacks=[CsvCallback(root=root)]
    )

    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)
    test_loader = create_loader(test_dataset, shuffle=shuffle)

    engine.eval(val_loader, tag='val', progressbar=progressbar, metrics=[metrics.Accuracy()])
    engine.eval(test_loader, progressbar=progressbar, metrics=[metrics.Accuracy()])
