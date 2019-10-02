import os
import random

import torch
from PIL import Image
from torchvision import transforms as transforms

from datasets import BaseDataset, get_transforms
from datasets.data_utils import (
    get_dir_file_extension,
    remove_top_dir,
    remove_extension,
    find_valid_files,
    decompress_cloth_segment,
    per_channel_transform,
    crop_tensors,
    get_norm_stats,
)


class WarpDataset(BaseDataset):
    """ Warp dataset for the warp module of SwapNet """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--input_transforms",
            nargs="+",
            default="none",
            choices=("none", "hflip", "vflip", "affine", "perspective", "all"),
            help="what random transforms to perform on the input ('all' for all transforms)",
        )
        if is_train:
            parser.set_defaults(
                input_transforms=("hflip", "vflip", "affine", "perspective")
            )
        parser.add_argument(
            "--per_channel_transform",
            action="store_true",
            default=True,  # TODO: make this a toggle based on if data is RGB or labels
            help="Perform the transform for each label instead of on the image as a "
            "whole. --cloth_representation must be 'labels'.",
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        self.cloth_dir = os.path.join(opt.dataroot, "cloth")
        extensions = [".npz"] if self.opt.cloth_representation is "labels" else None
        print("Extensions:", extensions)
        self.cloth_files = find_valid_files(self.cloth_dir, extensions)
        self.body_dir = os.path.join(opt.dataroot, "body")
        self.body_norm_stats = get_norm_stats(opt.dataroot, "body")
        opt.body_norm_stats = self.body_norm_stats
        self._normalize_body = transforms.Normalize(*self.body_norm_stats)

        self.crop_bounds = eval(opt.crop_bounds) if opt.crop_bounds else None
        self.cloth_transform = get_transforms(opt)

    def __len__(self):
        """
        Get the length of usable images. Note the length of cloth and body segmentations should be same
        """
        return len(self.cloth_files)

    def _load_cloth(self, index) -> torch.Tensor:
        """
        Loads the cloth file as a tensor
        """
        cloth_file = self.cloth_files[index]
        target_cloth_tensor = decompress_cloth_segment(
            cloth_file, self.opt.cloth_channels
        )
        if self.opt.dataset_mode == "image":
            # in image mode, the input cloth is the same as the target cloth
            input_cloth_tensor = target_cloth_tensor.clone()
        elif self.opt.dataset_mode == "video":
            # video mode, can choose a random image
            input_file = self.cloth_files[random.randint(0, len(self))]
            input_cloth_tensor = decompress_cloth_segment(
                input_file, self.opt.cloth_channels
            )
        return input_cloth_tensor, target_cloth_tensor

    def _load_body(self, index):
        """ Loads the body file as a tensor """
        cloth_file = self.cloth_files[index]
        body_file = get_corresponding_file(cloth_file, self.body_dir)
        as_pil_image = Image.open(body_file).convert("RGB")
        # TODO: normalize the image
        return self._normalize_body(transforms.ToTensor()(as_pil_image))

    def _perform_cloth_transform(self, cloth_tensor):
        """ Either does per-channel transform or whole-image transform """
        if self.opt.per_channel_transform:
            return per_channel_transform(cloth_tensor, self.cloth_transform)
        else:
            raise NotImplementedError("Sorry, per_channel_transform must be true")
            return self.input_transform(cloth_tensor)

    def __getitem__(self, index):
        """
        :returns:
            For training, return (input) AUGMENTED cloth seg, (input) body seg and (target) cloth seg
            of the SAME image
            For inference (e.g validation), return (input) cloth seg and (input) body seg
            of 2 different images
        """

        # the input cloth segmentation
        input_cloth_tensor, target_cloth_tensor = self._load_cloth(index)
        body_tensor = self._load_body(index)
        # apply the transformation for input cloth segmentation
        if self.cloth_transform:
            input_cloth_tensor = self._perform_cloth_transform(input_cloth_tensor)
        # crop to the proper image size
        if self.crop_bounds:
            input_cloth_tensor, target_cloth_tensor, body_tensor = crop_tensors(
                input_cloth_tensor,
                target_cloth_tensor,
                body_tensor,
                crop_bounds=self.crop_bounds,
            )

        return body_tensor, input_cloth_tensor, target_cloth_tensor


def get_corresponding_file(original, target_dir, target_ext=None):
    # number of top dir to replace
    num_top_parts = len(os.path.split(target_dir))
    # replace the top dirs
    top_removed = remove_top_dir(original, num_top_parts)
    target_file = os.path.join(target_dir, top_removed)
    # extension of files in the target dir
    if not target_ext:
        target_ext = get_dir_file_extension(target_dir)
    # change the extension
    target_file = remove_extension(target_file) + target_ext
    return target_file
