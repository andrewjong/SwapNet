"""
A list of base options common to all stages
"""
import copy
import sys
import argparse
import json
import os

import torch

import datasets
import models
import optimizers
from util.util import PromptOnce

datasets, models, optimizers  # so auto import doesn't remove above


class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            conflict_handler="resolve",
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
        parser.add_argument(
            "--comments",
            default="",
            help="additional comments to add to this experiment, saved in args.json",
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
            "--model", help="which model to run", choices=("warp", "texture", "pix2pix")
        )
        parser.add_argument(
            "--checkpoints_dir", default="./checkpoints", help="Where to save models"
        )
        parser.add_argument(
            "--load_epoch",
            default="latest",
            help="epoch to load (use with --continue_train or for inference, 'latest' "
                 "for latest ",
        )
        # == DATA / IMAGE LOADING ==
        parser.add_argument(
            "--dataroot",
            required=True,
            help="path to data, should contain 'cloth/', 'body/', 'texture/', "
                 "'rois.csv'",
        )
        parser.add_argument(
            "--dataset", help="dataset class to use, if none then will use model name"
        )
        parser.add_argument(
            "--dataset_mode",
            default="image",
            choices=("image", "video"),
            help="how data is formatted. video mode allows additional source inputs"
                 "from other frames of the video",
        )
        # channels
        parser.add_argument(
            "--cloth_representation",
            default="labels",  # default according to SwapNet
            choices=("rgb", "labels"),
            help="which representation the cloth segmentations are in. 'labels' means "
                 "a 2D tensor where each value is the cloth label. 'rgb' ",
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
        parser.add_argument(
            "--crop_size", type=int, default=128, help="then crop to this size"
        )
        parser.add_argument(
            "--crop_bounds",
            help="DO NOT USE WITH --crop_size. crop images to a region: ((xmin, ymin), (xmax, ymax))",
        )
        # == ITERATION PROPERTIES ==
        parser.add_argument(
            "--max_dataset_size", type=int, default=float("inf"), help="cap on data"
        )
        parser.add_argument(
            "--batch_size", type=int, default=8, help="batch size to load data"
        )
        parser.add_argument(
            "--shuffle_data",
            default=True,
            type=bool,
            help="whether to shuffle dataset (default is True)",
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
        parser.add_argument(
            "--no_confirm", action="store_true", help="do not prompt for confirmations"
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
        opt.batch_size

        # modify options for each arg that can do so
        modifiers = ["model", "dataset"]
        if self.is_train:
            modifiers.append("optimizer_D")
        for arg in modifiers:
            # becomes model(s), dataset(s), optimizer(s)
            import_source = eval(arg.split("_")[0] + "s")
            # becomes e.g. opt.model, opt.dataset, opt.optimizer
            name = getattr(opt, arg)
            print(arg, name)
            if name is not None:
                options_modifier = import_source.get_options_modifier(name)
                parser = options_modifier(parser, self.is_train)
                opt, _ = parser.parse_known_args()
            # hacky, add optimizer G params if different from opt_D
            if arg is "optimizer_D" and opt.optimizer_D != opt.optimizer_G:
                modifiers.append("optimizer_G")

        self._parser = parser
        final_opt = self._parser.parse_args()
        return final_opt

    @staticmethod
    def _validate(opt):
        """
        Validate that options are correct
        :return:
        """
        assert (
                opt.crop_size <= opt.load_size
        ), "Crop size must be less than or equal to load size "

    def parse(self, print_options=True, store_options=True, user_overrides=True):
        """

        Args:
            print_options: print the options to screen when parsed
            store_options: save the arguments to file: "{opt.checkpoints_dir}/{opt.name}/args.json"

        Returns:

        """
        opt = self.gather_options()
        opt.is_train = self.is_train

        # perform assertions on arguments
        BaseOptions._validate(opt)

        if opt.gpu_id > 0:
            torch.cuda.set_device(opt.gpu_id)
            torch.backends.cudnn.benchmark = True

        self.opt = opt

        # Load options from config file if present
        if opt.config_file:
            self.load(opt.config_file, user_overrides)

        if print_options:  # print what we parsed
            self.print()

        root = opt.checkpoints_dir if self.is_train else opt.results_dir
        self.save_file = os.path.join(root, opt.name, "args.json")
        if store_options:  # store options to file
            self.save()
        return opt

    def print(self):
        """
        prints the options nicely
        :return:
        """
        d = vars(self.opt)
        print("=====OPTIONS======")
        for k, v in d.items():
            print(k, ":", v)
        print("==================")

    def save(self):
        """
        Saves to a .json file
        :return:
        """
        d = vars(self.opt)

        PromptOnce.makedirs(os.path.dirname(self.save_file), not self.opt.no_confirm)
        with open(self.save_file, "w") as f:
            f.write(json.dumps(d, indent=4))

    def load(self, json_file, user_overrides):
        load(self.opt, json_file, user_overrides=user_overrides)


def load(opt, json_file, user_overrides=True):
    """

    Args:
        opt: Namespace that will get modified
        json_file:
        user_overrides: whether user command line arguments should override anything being loaded from the config file

    """
    opt = copy.deepcopy(opt)
    with open(json_file, "r") as f:
        args = json.load(f)

    # if the user specifies arguments on the command line, don't override these
    if user_overrides:
        user_args = filter(lambda a: a.startswith("--"), sys.argv[1:])
        user_args = set(
            [a.lstrip("-") for a in user_args]
        )  # get rid of left dashes
        print("Not overriding:", user_args)

    # override default options with values in config file
    for k, v in args.items():
        # only override if not specified on the cmdline
        if not user_overrides or (user_overrides and k not in user_args):
            setattr(opt, k, v)
    # but make sure the config file matches up
    opt.config_file = json_file
    return opt
