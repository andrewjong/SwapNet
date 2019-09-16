import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as transforms

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


class TextureDataset(BaseDataset):
    """ Texture dataset for the texture module of SwapNet """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.set_defaults(input_transforms=("h_flip", "v_flip"))
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        # get all texture files
        self.texture_dir = os.path.join(opt.dataroot, "texture")
        self.texture_files = find_valid_files(self.texture_dir, IMG_EXTENSIONS)

        self.texture_norm_stats = get_norm_stats(opt.dataroot, "texture")
        opt.texture_norm_stats = self.texture_norm_stats
        self._normalize_texture = transforms.Normalize(*self.texture_norm_stats)

        # cloth files
        self.cloth_dir = os.path.join(opt.dataroot, "cloth")
        self.cloth_ext = get_dir_file_extension(self.cloth_dir)

        self.rois_db = os.path.join(opt.dataroot, "rois.csv")
        self.rois_df = pd.read_csv(self.rois_db, index_col=0)
        # todo: remove None values preemptively, else we have to fill in with 0
        self.rois_df = self.rois_df.replace("None", 0).astype(np.float32)

        self.crop_bounds = eval(opt.crop_bounds) if opt.crop_bounds else None

        self.input_transform = get_transforms(opt)

    def __len__(self):
        return len(self.texture_files)

    def __getitem__(self, index: int):
        """ """
        # (1) Get target texture.
        target_texture_file = self.texture_files[index]
        target_texture_img = Image.open(target_texture_file).convert("RGB")
        target_texture_tensor = self._normalize_texture(
            transforms.ToTensor()(target_texture_img)
        )

        # file id
        file_id = remove_extension(target_texture_file).lstrip(self.texture_dir + "/")

        # (2) Get corresponding cloth.
        cloth_file = os.path.join(self.cloth_dir, file_id + self.cloth_ext)
        cloth_tensor = decompress_cloth_segment(cloth_file, n_labels=19)

        # (3) Get corresponding roi.
        rois = self.rois_df.loc[file_id].values
        rois_tensor = torch.from_numpy(rois)

        # (4) Get randomly flipped input.
        # input will be randomly flipped of target; if we flip input, we must flip rois
        input_texture_image, rois_tensor = random_image_roi_flip(
            target_texture_img, rois_tensor
        )
        input_texture_tensor = self._normalize_texture(
            transforms.ToTensor()(input_texture_image)
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

        return input_texture_tensor, rois_tensor, cloth_tensor, target_texture_tensor