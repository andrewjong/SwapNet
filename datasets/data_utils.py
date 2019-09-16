import os
import pandas as pd
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image
from scipy.sparse import load_npz
from torch import Tensor
from torchvision.transforms import functional as TF

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]
NP_EXTENSIONS = [".npz"]  # numpy compressed


def get_norm_stats(dataroot, key):
    df = pd.read_json(
        os.path.join(dataroot, "normalization_stats.json"), lines=True
    ).set_index("path")
    series = df.loc[key]
    return series["means"], series["stds"]

def unnormalize(tensor, mean, std, clamp=True, inplace=False):
    if not inplace:
        tensor = tensor.clone()

    def unnormalize_1(ten, men, st):
        for t, m, s in zip(ten, men, st):
            t.mul_(s).add_(m)
            if clamp:
                t.clamp_(0, 1)

    if tensor.shape == 4:
        # then we have batch size in front or something
        for t in tensor:
            unnormalize_1(t, mean, std)
    else:
        unnormalize_1(tensor, mean, std)

    return tensor

def scale_tensor(tensor, scale_each=False, range=None):
    """
    From torchvision's make_grid
    :return:
    """
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if range is not None:
        assert isinstance(range, tuple), \
            "range has to be a tuple (min, max) if specified. min and max are numbers"

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        for t in tensor:  # loop over mini-batch dimension
            norm_range(t, range)
    else:
        norm_range(tensor, range)

    return tensor


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def in_extensions(filename, extensions):
    return any(filename.endswith(extension) for extension in extensions)


def find_valid_files(dir, extensions, max_dataset_size=float("inf")):
    """
    Get all the images recursively under a dir.
    Args:
        dir:
        max_dataset_size:

    Returns: found files, where each item is a tuple (id, ext)

    """
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if in_extensions(fname, extensions):
                path = os.path.join(root, fname)
                images.append(path)
    return images[: min(max_dataset_size, len(images))]


def get_dir_file_extension(dir, check=5):
    """
    Guess what extensions are for all files in a dir.
    Args:
        dir:
        check:

    Returns:

    """
    exts = []
    for root, _, fnames in os.walk(dir):
        for fname in fnames[:check]:
            ext = os.path.splitext(fname)[1]
            exts.append(ext)

    return Counter(exts).most_common(1)[0][0]


def remove_top_dir(dir, n=1):
    """
    Removes the top level dirs from a path
    Args:
        dir:
        n:

    Returns:

    """
    parts = dir.split(os.path.sep)
    top_removed = os.path.sep.join(parts[n + 1 :])
    return top_removed


def remove_extension(fname):
    return os.path.splitext(fname)[0]


def change_extension(fname, ext1, ext2):
    """
    :return: file name with new extension
    """
    return fname[: -len(ext1)] + ext2


def crop_tensors(*tensors, crop_bounds=((0, -1), (0, -1))):
    """
    Crop multiple tensors
    Args:
        *tensors:
        crop_bounds:

    Returns:

    """
    ret = []
    for t in tensors:
        ret.append(crop_tensor(t, crop_bounds))
    return ret


def crop_tensor(tensor: Tensor, crop_bounds):
    """
    Crops a tensor at the given crop bounds.
    :param tensor:
    :param crop_bounds: ((h_min, h_max), (w_min,w_max))
    :return:
    """
    (h_min, h_max), (w_min, w_max) = crop_bounds
    return tensor[:, h_min:h_max, w_min:w_max]


def crop_rois(rois: np.ndarray, crop_bounds):
    # TODO: might have to worry about nan values?
    if crop_bounds is not None:
        rois = rois.copy()
        (h_min, h_max), (w_min, w_max) = crop_bounds
        # clip the x-axis to be within bounds. xmin and xmax index
        xs = rois[:, (1, 2)]
        xs = np.clip(xs, w_min, w_max - 1)
        xs -= xs.min(axis=0)  # translate
        # clip the y-axis to be within bounds. ymin and ymax index
        ys = rois[:, (3, 4)]
        ys = np.clip(ys, h_min, h_max - 1)
        ys -= ys.min(axis=0)  # translate
        # put it back together again
        rois = np.stack((rois[:, 0], xs[:, 0], ys[:, 0], xs[:, 1], ys[:, 1]))
        # transpose because stack stacked them opposite of what we want
        rois = rois.T
    return rois


def random_image_roi_flip(img, rois, vp=0.5, hp=0.5):
    """
    Randomly flips an image and associated ROI tensor together.
    I.e. if the image flips, the ROI will flip to match.
    Args:
        img: a PIL image
        rois:
        vp:
        hp:
    Returns: flipped PIL image, flipped rois
    """
    W, H = img.size

    if random.random() < vp:
        img = TF.vflip(img)
        center = int(H / 2)
        flip_rois_(rois, 0, center)

    if random.random() < hp:
        img = TF.hflip(img)
        center = int(W / 2)
        flip_rois_(rois, 1, center)

    return img, rois


def flip_rois_(rois, axis, center):
    """
    Flips rois in place, along the given axis, at the given center value
    Args:
        rois: roi tensor
        axis: 0 for a vertical flip, 1 for a horizontal flip
        center: a positive integer, where to flip
    E.g. if axis=1
    ------------       ------------
    |    |     |       |    |     |
    | +  |     |  =>   |    |   + |
    |    |     |       |    |     |
    |    |     |       |    |     |
    ------------       ------------
    Returns:
    """
    if axis == 0:  # vertical flip, flip y values
        min_idx, max_idx = -3, -1  # use negative indexing in case of batch in 1st dim
    elif axis == 1:  # horizontal flip, flip x values
        min_idx, max_idx = -4, -2
    else:
        raise ValueError(f"dim argument must be 0 or 1, received {axis}")

    # put side by side
    values = torch.stack((rois[:, min_idx], rois[:, max_idx]))
    # compute the flip
    delta = center - values
    flipped = center + delta
    # max and min are now swapped because we flipped
    max, min = torch.chunk(flipped, 2)
    # put them back in
    rois[:, min_idx], rois[:, max_idx] = min, max


def decompress_cloth_segment(fname, n_labels) -> Tensor:
    """
    Load cloth segmentation sparse matrix npz file and output a one hot tensor.
    :return: tensor of size(H,W,n_labels)
    """
    try:
        data_sparse = load_npz(fname)
    except Exception as e:
        print("Could not decompress cloth segment:", fname)
        raise e
    return to_onehot_tensor(data_sparse, n_labels)


def to_onehot_tensor(sp_matrix, n_labels):
    """
    convert sparse scipy labels matrix to onehot pt tensor of size (n_labels,H,W)
    Note: sparse tensors aren't supported in multiprocessing https://github.com/pytorch/pytorch/issues/20248

    :param sp_matrix: sparse 2d scipy matrix, with entries in range(n_labels)
    :return: pt tensor of size(n_labels,H,W)
    """
    sp_matrix = sp_matrix.tocoo()
    indices = np.vstack((sp_matrix.data, sp_matrix.row, sp_matrix.col))
    indices = torch.LongTensor(indices)
    values = torch.Tensor([1.0] * sp_matrix.nnz)
    shape = (n_labels,) + sp_matrix.shape
    return torch.sparse.FloatTensor(indices, values, torch.Size(shape)).to_dense()


def per_channel_transform(input_tensor, transform_function) -> Tensor:
    """
    Randomly transform each of n_channels of input data.
    Out of place operation
    :param input_tensor: must be a numpy array of size (n_channels, w, h)
    :param transform_function: any torchvision transforms class
    :return: transformed pt tensor
    """
    input_tensor = input_tensor.numpy()
    tform_input_np = np.zeros(shape=input_tensor.shape, dtype=input_tensor.dtype)
    n_channels = input_tensor.shape[0]
    for i in range(n_channels):
        tform_input_np[i] = np.array(
            transform_function(Image.fromarray(input_tensor[i]))
        )
    return torch.from_numpy(tform_input_np)