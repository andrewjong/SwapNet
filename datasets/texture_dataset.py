import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as transforms
from torchvision.transforms import functional as tf

from datasets import BaseDataset, get_transforms
from datasets.data_utils import (
    IMG_EXTENSIONS,
    find_valid_files,
    get_dir_file_extension,
    remove_extension,
    decompress_cloth_segment,
    random_image_roi_flip,
    crop_tensors,
    crop_rois,
    get_norm_stats,
)
from util.util import remove_prefix


class TextureDataset(BaseDataset):
    """ Texture dataset for the texture module of SwapNet """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # transforms
        parser.add_argument(
            "--input_transforms",
            nargs="+",
            default="none",
            choices=("none", "hflip", "vflip", "all"),
            help="what random transforms to perform on the input ('all' for all transforms)",
        )
        if is_train:
            parser.set_defaults(input_transforms=("hflip", "vflip"))
        return parser

    def __init__(self, opt, texture_dir=None, cloth_dir=None):
        """

        Args:
            opt: Namespace object
            texture_dir (str): optional override path to texture dir
            cloth_dir (str): optional override path to cloth dir
        """
        super().__init__(opt)
        # get all texture files
        self.texture_dir = (
            texture_dir if texture_dir else os.path.join(opt.dataroot, "texture")
        )
        self.texture_files = find_valid_files(self.texture_dir, IMG_EXTENSIONS)

        self.texture_norm_stats = get_norm_stats(
            os.path.dirname(self.texture_dir), "texture"
        )
        opt.texture_norm_stats = self.texture_norm_stats
        self._normalize_texture = transforms.Normalize(*self.texture_norm_stats)

        # cloth files
        self.cloth_dir = cloth_dir if cloth_dir else os.path.join(opt.dataroot, "cloth")
        self.cloth_ext = get_dir_file_extension(self.cloth_dir)
        if not self.is_train:
            self.cloth_files = find_valid_files(self.cloth_dir, extensions=".npz")
            if not opt.shuffle_data:
                self.cloth_files.sort()

        # load rois
        self.rois_db = os.path.join(opt.dataroot, "rois.csv")
        self.rois_df = pd.read_csv(self.rois_db, index_col=0)
        # todo: remove None values preemptively, else we have to fill in with 0
        self.rois_df = self.rois_df.replace("None", 0).astype(np.float32)

        # # per-channel transforms on the input
        # self.input_transform = get_transforms(opt)

    def __len__(self):
        if self.is_train:
            return len(self.texture_files)
        else:
            return min(len(self.texture_files), len(self.cloth_files))

    def __getitem__(self, index: int):
        """ """
        # (1) Get target texture.
        target_texture_file = self.texture_files[index]
        target_texture_img = Image.open(target_texture_file).convert("RGB")

        target_texture_tensor = self._normalize_texture(
            tf.to_tensor(tf.resize(target_texture_img, self.opt.load_size))
        )

        # file id for matching cloth and matching ROI
        file_id = remove_prefix(
            remove_extension(target_texture_file), self.texture_dir + "/"
        )

        # (2) Get corresponding cloth if train, else cloth at index if inference.
        cloth_file = (
            os.path.join(self.cloth_dir, file_id + self.cloth_ext)
            if self.is_train
            else self.cloth_files[index]
        )
        cloth_tensor = decompress_cloth_segment(cloth_file, n_labels=19)
        # resize cloth tensor
        # We have to unsqueeze because interpolate expects batch in dim1
        cloth_tensor = nn.functional.interpolate(
            cloth_tensor.unsqueeze(0), size=self.opt.load_size
        ).squeeze()

        # (3) Get and scale corresponding roi.
        original_size = target_texture_img.size[0]  # PIL width
        scale = float(self.opt.load_size) / original_size
        rois = np.rint(self.rois_df.loc[file_id].values * scale)
        rois_tensor = torch.from_numpy(rois)

        # (4) Get randomly flipped input.
        # input will be randomly flipped of target; if we flip input, we must flip rois
        hflip = (
            0.5 if any(t in self.opt.input_transforms for t in ("hflip", "all")) else 0
        )
        vflip = (
            0.5 if any(t in self.opt.input_transforms for t in ("vflip", "all")) else 0
        )
        input_texture_image, rois_tensor = random_image_roi_flip(
            target_texture_img, rois_tensor, vp=vflip, hp=hflip
        )
        input_texture_tensor = self._normalize_texture(
            tf.to_tensor(tf.resize(input_texture_image, self.opt.load_size))
        )

        # do cropping if needed
        if self.crop_bounds:
            input_texture_tensor, cloth_tensor, target_texture_tensor = crop_tensors(
                input_texture_tensor,
                cloth_tensor,
                target_texture_tensor,
                crop_bounds=self.crop_bounds,
            )
            rois_tensor = crop_rois(rois_tensor, self.crop_bounds)

        # assert shapes
        assert (
            input_texture_tensor.shape[-2:]
            == target_texture_tensor.shape[-2:]
            == cloth_tensor.shape[-2:]
        ), f"input {input_texture_tensor.shape}; target {target_texture_tensor.shape}; cloth {cloth_tensor.shape}"

        return {
            "texture_paths": target_texture_file,
            "input_textures": input_texture_tensor,
            "rois": rois_tensor,
            "cloth_paths": cloth_file,
            "cloths": cloth_tensor,
            "target_textures": target_texture_tensor,
        }
