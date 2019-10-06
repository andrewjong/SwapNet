from abc import ABC, abstractmethod

import torch.utils.data as data


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        self.crop_bounds = self.parse_crop_bounds()
        self.is_train = opt.is_train

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    def parse_crop_bounds(self):
        if isinstance(self.opt.crop_size, int) and self.opt.crop_size < self.opt.load_size:
            minimum = int((self.opt.load_size - self.opt.crop_size) / 2)
            maximum = self.opt.load_size - minimum
            crop_bounds = (minimum, minimum), (maximum, maximum) # assuming square
        else:
            crop_bounds = eval(self.opt.crop_bounds) if self.opt.crop_bounds else None
        return crop_bounds
