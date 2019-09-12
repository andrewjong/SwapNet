"""
A list of base options common to all stages
"""
import argparse

import torch

import datasets
import models
import optimizers

datasets, models, optimizers  # so auto import doesn't remove above


class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        # == EXPERIMENT SETUP ==
        parser.add_argument(
            "--config_file",
            help="load arguments from a json file instead of command line",
        )
        parser.add_argument(
            "--name",
            default="my_experiment",
            help="name of the experiment, determines where things are saved",
        )
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument(
            "--display_winsize",
            type=int,
            default=256,
            help="display window size for both visdom and HTML",
        )
        # == MODEL INIT / LOADING / SAVING ==
        parser.add_argument(
            "--model",
            help="which model to run",
            choices=("warp", "texture"),
            required=True,
        )
        parser.add_argument(
            "--checkpoints_dir", default="./checkpoints", help="Where to save models"
        )
        parser.add_argument(
            "--from_epoch", default="latest", help="epoch to load, 'latest' for latest"
        )
        # == DATA / IMAGE LOADING ==
        parser.add_argument(
            "--dataroot",
            required=True,
            help="path to data, should contain 'cloth/', 'body/', 'texture/', "
            "'rois.csv'",
        )
        parser.add_argument(
            "--dataset_mode",
            default="image",
            choices=("image", "video"),
            help="how data is formatted",
        )
        # channels
        parser.add_argument(
            "--cloth_representation",
            default="labels",  # default according to SwapNet
            choices=("rgb", "labels"),
            help="which representation the cloth segmentations are in. 'labels' means a 2D tensor where each value is the cloth label. 'rgb' ",
        )
        parser.add_argument(
            "--body_representation",
            default="rgb",  # default according to SwapNet
            choices=("rgb", "labels"),
            help="which representation the body segmentations are in",
        )
        parser.add_argument(
            "--cloth_channels",
            default=19,
            type=int,
            help="only used if --cloth_representation == 'labels'. cloth segmentation "
            "number of channels",
        )
        parser.add_argument(
            "--body_channels",
            default=12,
            type=int,
            help="only used if --body_representation == 'labels'. body segmentation "
            "number of channels. Use 12 for neural body fitting output",
        )
        parser.add_argument(
            "--texture_channels",
            default=3,
            type=int,
            help="RGB textured image number of channels",
        )
        # image dimension / editing
        parser.add_argument(
            "--pad", action="store_true", help="add a padding to make image square"
        )
        parser.add_argument(
            "--load_size",
            default=128,
            type=int,
            help="scale images (after padding) to this size",
        )
        parser.add_argument("--crop_size", default=128, help="then crop to this size")
        parser.add_argument(
            "--crop_bounds",
            help="DO NOT USE WITH --crop_size. crop images to a region: ((hmin, hmax), (wmin, wmax))",
        )
        # transforms
        parser.add_argument(
            "--input_transforms",
            nargs="+",
            choices=("none", "h_flip", "v_flip", "affine", "perspective", "all"),
            help="what random transforms to perform on the input ('all' for all transforms)",
        )
        # == ITERATION PROPERTIES ==
        parser.add_argument(
            "--max_dataset_size", type=int, default=float("inf"), help="cap on data"
        )
        parser.add_argument(
            "--batch_size", type=int, default=4, help="batch size to load data"
        )
        parser.add_argument(
            "--shuffle_data",
            action="store_true",
            help="whether to shuffle dataset (default is No)",
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="number of CPU threads for data loading",
        )
        parser.add_argument(
            "--gpu_id", default=0, type=int, help="gpu id to use. -1 for cpu"
        )

        self._parser = parser
        self.is_train = None

    def gather_options(self):
        """
        Gathers options from all modifieable thingies.
        :return:
        """
        parser = self._parser

        # basic options
        opt, _ = parser.parse_known_args()
        parser.set_defaults(dataset=opt.model)

        # modify options for each arg that can do so
        modifiers = ["model", "dataset", "optimizer_D"]
        for arg in modifiers:
            # becomes model(s), dataset(s), optimizer(s)
            import_source = eval(arg.split("_")[0] + "s")
            # becomes e.g. opt.model, opt.dataset, opt.optimizer
            name = getattr(opt, arg)
            options_modifier = import_source.get_options_modifier(name)
            parser = options_modifier(parser, self.is_train)
            opt, _ = parser.parse_known_args()
            # hacky, add optimizer G params if different from opt_D
            if arg is "optimizer_D" and opt.optimizer_D != opt.optimizer_G:
                modifiers.append("optimizer_G")

        self._parser = parser
        final_opt = self._parser.parse_args()
        return final_opt

    def parse(self):
        opt = self.gather_options()
        opt.is_train = self.is_train

        if opt.gpu_id > 0:
            torch.cuda.set_device(opt.gpu_id)
            torch.backends.cudnn.benchmark = True

        self.opt = opt
        return opt
