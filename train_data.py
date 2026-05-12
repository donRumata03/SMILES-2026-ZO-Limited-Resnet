import os

from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY=True


def _balanced_indices(targets):
    counts = [0] * 100
    indices = []
    for index, target in enumerate(targets):
        if counts[target] < 81:
            counts[target] += 1
            indices.append(index)
        if len(indices) == 8100:
            break
    return indices


def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,

):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    # a hack since it's unclear if the parameters of the head init function are allowed to be changed
    # this would guarantee dataset dir transfer so we could use the dataset for a more meaningful initialization
    os.environ["CIFAR100_DATA_DIR"] = data_dir
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY, # True
        download=True,
        transform=get_transforms(train=True),
    )
    train_dataset = Subset(train_dataset, _balanced_indices(train_dataset.targets))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train
    )

    return train_dataset, train_loader
