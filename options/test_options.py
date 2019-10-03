import argparse

from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self, **defaults):
        super().__init__()
        self.is_train = False
        parser = self._parser

        parser.add_argument(
            "--warp_checkpoint",
            help="checkpoint file of warp stage model, containing args.json file in "
            "same dir",
        )
        parser.add_argument(
            "--texture_checkpoint",
            help="checkpoint dir of texture stage containing args.json file",
        )
        parser.add_argument(
            "--cloth_dir",
            required=True,
            help="Root directory to use for the clothing source. If same directory as "
            "--body_root, use --shuffle_data to achieve clothing transfer",
        )
        parser.add_argument(
            "--body_dir",
            required=True,
            help="Root directory to use as target bodys for where the cloth will be placed "
            "on. If same directory as --cloth_root, use --shuffle_data to achieve "
            "clothing transfer",
        )
        parser.add_argument(
            "--results_dir",
            default="results",
            help="folder to output intermediate and final results",
        )
        parser.add_argument("--compute_intermediates", action="store_true", help="compute and save intermediate visuals")

        # remove arguments
        parser.add_argument("--dataroot", help=argparse.SUPPRESS)  # remove dataroot arg
        parser.add_argument("--model", help=argparse.SUPPRESS)  # remove model as we restore from checkpoint

        parser.set_defaults(**defaults)
