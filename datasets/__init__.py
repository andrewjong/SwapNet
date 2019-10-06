import importlib

import torch
from torchvision.transforms import transforms

from datasets.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            f"In {dataset_filename}.py, there should be a subclass of BaseDataset "
            f"with class name that matches {target_dataset_name} in lowercase."
        )

    return dataset

def get_options_modifier(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, **ds_kwargs):
    """Create a dataset given the option.

    This function wraps the class CappedDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from datasets import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CappedDataLoader(opt, **ds_kwargs)
    return data_loader


class CappedDataLoader:
    """Wrapper class of Dataset class that caps the data limit at the specified
    max_dataset_size """

    def __init__(self, opt, **ds_kwargs):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dname = opt.dataset if opt.dataset else opt.model
        print(f"Creating dataset {dname}...", end=" ")
        dataset_class = find_dataset_using_name(dname)
        self.dataset = dataset_class(opt, **ds_kwargs)
        print(f"dataset [{type(self.dataset).__name__}] was created")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.shuffle_data,
            num_workers=opt.num_workers,
        )

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data


def get_transforms(opt):
    """
    Return Composed torchvision transforms based on specified arguments.
    """
    transforms_list = []
    if "none" in opt.input_transforms:
        return
    every = "all" in opt.input_transforms

    if every or "vflip" in opt.input_transforms:
        transforms_list.append(transforms.RandomVerticalFlip())
    if every or "hflip" in opt.input_transforms:
        transforms_list.append(transforms.RandomHorizontalFlip())
    if every or "affine" in opt.input_transforms:
        transforms_list.append(
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=20
            )
        )
    if every or "perspective" in opt.input_transforms:
        transforms_list.append(transforms.RandomPerspective())

    return transforms.RandomOrder(transforms_list)
