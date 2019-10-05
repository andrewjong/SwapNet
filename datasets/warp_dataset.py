import os
import random
from typing import Tuple

from torch import Tensor
from PIL import Image
from torch import nn
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

    def __init__(self, opt, cloth_dir=None, body_dir=None):
        """

        Args:
            opt:
            cloth_dir: (optional) path to cloth dir, if provided
            body_dir: (optional) path to body dir, if provided
        """
        super().__init__(opt)

        self.cloth_dir = cloth_dir if cloth_dir else os.path.join(opt.dataroot, "cloth")
        print("cloth dir", self.cloth_dir)
        extensions = [".npz"] if self.opt.cloth_representation == "labels" else None
        print("Extensions:", extensions)
        self.cloth_files = find_valid_files(self.cloth_dir, extensions)
        if not opt.shuffle_data:
            self.cloth_files.sort()

        self.body_dir = body_dir if body_dir else os.path.join(opt.dataroot, "body")
        if not self.is_train:  # only load these during inference
            self.body_files = find_valid_files(self.body_dir)
            if not opt.shuffle_data:
                self.body_files.sort()
        print("body dir", self.body_dir)
        self.body_norm_stats = get_norm_stats(os.path.dirname(self.body_dir), "body")
        opt.body_norm_stats = self.body_norm_stats
        self._normalize_body = transforms.Normalize(*self.body_norm_stats)

        self.cloth_transform = get_transforms(opt)

    def __len__(self):
        """
        Get the length of usable images. Note the length of cloth and body segmentations should be same
        """
        if not self.is_train:
            return min(len(self.cloth_files), len(self.body_files))
        else:
            return len(self.cloth_files)

    def _load_cloth(self, index) -> Tuple[str, Tensor, Tensor]:
        """
        Loads the cloth file as a tensor
        """
        cloth_file = self.cloth_files[index]
        target_cloth_tensor = decompress_cloth_segment(
            cloth_file, self.opt.cloth_channels
        )
        if self.is_train:
            # during train, we want to do some fancy transforms
            if self.opt.dataset_mode == "image":
                # in image mode, the input cloth is the same as the target cloth
                input_cloth_tensor = target_cloth_tensor.clone()
            elif self.opt.dataset_mode == "video":
                # video mode, can choose a random image
                cloth_file = self.cloth_files[random.randint(0, len(self)) - 1]
                input_cloth_tensor = decompress_cloth_segment(
                    cloth_file, self.opt.cloth_channels
                )
            else:
                raise ValueError(self.opt.dataset_mode)

            # apply the transformation for input cloth segmentation
            if self.cloth_transform:
                input_cloth_tensor = self._perform_cloth_transform(input_cloth_tensor)

            return cloth_file, input_cloth_tensor, target_cloth_tensor
        else:
            # during inference, we just want to load the current cloth
            return cloth_file, target_cloth_tensor, target_cloth_tensor

    def _load_body(self, index):
        """ Loads the body file as a tensor """
        if self.is_train:
            # use corresponding strategy during train
            cloth_file = self.cloth_files[index]
            body_file = get_corresponding_file(cloth_file, self.body_dir)
        else:
            # else we have to load by index
            body_file = self.body_files[index]
        as_pil_image = Image.open(body_file).convert("RGB")
        body_tensor = self._normalize_body(transforms.ToTensor()(as_pil_image))
        return body_file, body_tensor

    def _perform_cloth_transform(self, cloth_tensor):
        """ Either does per-channel transform or whole-image transform """
        if self.opt.per_channel_transform:
            return per_channel_transform(cloth_tensor, self.cloth_transform)
        else:
            raise NotImplementedError("Sorry, per_channel_transform must be true")
            # return self.input_transform(cloth_tensor)

    def __getitem__(self, index):
        """
        :returns:
            For training, return (input) AUGMENTED cloth seg, (input) body seg and (target) cloth seg
            of the SAME image
            For inference (e.g validation), return (input) cloth seg and (input) body seg
            of 2 different images
        """

        # the input cloth segmentation
        cloth_file, input_cloth_tensor, target_cloth_tensor = self._load_cloth(index)
        body_file, body_tensor = self._load_body(index)

        # RESIZE TENSORS
        # We have to unsqueeze because interpolate expects batch in dim1
        input_cloth_tensor = nn.functional.interpolate(
            input_cloth_tensor.unsqueeze(0), size=self.opt.load_size
        ).squeeze()
        if self.is_train:
            target_cloth_tensor = nn.functional.interpolate(
                target_cloth_tensor.unsqueeze(0), size=self.opt.load_size
            ).squeeze()
        body_tensor = nn.functional.interpolate(
            body_tensor.unsqueeze(0),
            size=self.opt.load_size,
            mode="bilinear",  # same as default for torchvision.resize
        ).squeeze()

        # crop to the proper image size
        if self.crop_bounds:
            input_cloth_tensor, body_tensor = crop_tensors(
                input_cloth_tensor, body_tensor, crop_bounds=self.crop_bounds
            )
            if self.is_train: # avoid extra work if we don't need targets for inference
                target_cloth_tensor = crop_tensors(
                    target_cloth_tensor, crop_bounds=self.crop_bounds
                )

        return {
            "body_paths": body_file,
            "bodys": body_tensor,
            "cloth_paths": cloth_file,
            "input_cloths": input_cloth_tensor,
            "target_cloths": target_cloth_tensor,
        }


def get_corresponding_file(original, target_dir, target_ext=None):
    """
    Say an original file is
        dataroot/subject/body/SAMPLE_ID.jpg

    And we want the corresponding file
        dataroot/subject/cloth/SAMPLE_ID.npz

    The corresponding file is in target_dir dataroot/subject/cloth, so we replace the
    top level directories with the target dir

    Args:
        original:
        target_dir:
        target_ext:

    Returns:

    """
    # number of top dir to replace
    num_top_parts = len(target_dir.split(os.path.sep))
    # replace the top dirs
    top_removed = remove_top_dir(original, num_top_parts)
    target_file = os.path.join(target_dir, top_removed)
    # extension of files in the target dir
    if not target_ext:
        target_ext = get_dir_file_extension(target_dir)
    # change the extension
    target_file = remove_extension(target_file) + target_ext
    return target_file
