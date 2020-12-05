from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset


def split_indices(n: int, **kwargs):
    return train_test_split(list(range(n)), **kwargs)


class Subset(VisionDataset):

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        super(Subset, self).__init__(None, transforms=None, transform=transform, target_transform=target_transform)
        del self.root

        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        x, y = self.dataset[self.indices[index]]

        if self.transforms:
            x, y = self.transforms(x, y)

        return x, y

    def __len__(self):
        return len(self.indices)
