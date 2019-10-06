import argparse

from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self, **defaults):
        super().__init__()
        self.is_train = False
        parser = self._parser

        parser.set_defaults(max_dataset_size=50, shuffle_data=False)
        parser.add_argument(
            "--interval",
            metavar="N",
            default=1,
            type=int,
            help="only run every n images",
        )
        parser.add_argument(
            "--warp_checkpoint",
            help="Use this to run the warp stage. Specifies the checkpoint file of "
            "warp stage model, containing args.json file in same dir",
        )
        parser.add_argument(
            "--texture_checkpoint",
            help="Use this to run the texture stage. Specifies the checkpoint dir of "
            "texture stage containing args.json file",
        )
        parser.add_argument(
            "--body_dir",
            required=False,  # don't require in case only running texture stage
            help="Directory to use as target bodys for where the cloth will be placed "
            "on. If same directory as --cloth_root, use --shuffle_data to achieve "
            "clothing transfer",
        )
        parser.add_argument(
            "--cloth_dir",
            required=True,
            help="Directory to use for the clothing source. If same directory as "
            "--body_root, use --shuffle_data to achieve clothing transfer",
        )
        parser.add_argument(
            "--texture_dir",
            required=False,  # don't require in case only running warp stage
            help="Directory to use for the clothing source. If same directory as "
            "--body_root, use --shuffle_data to achieve clothing transfer",
        )
        parser.add_argument(
            "--results_dir",
            default="results",
            help="folder to output intermediate and final results",
        )
        parser.add_argument(
            "--skip_intermediates",
            action="store_true",
            help="choose not to save intermediate cloth visuals as images for warp "
            "stage (instead, just save .npz files)",
        )

        # remove arguments
        parser.add_argument("--dataroot", help=argparse.SUPPRESS)  # remove dataroot arg
        parser.add_argument(
            "--model", help=argparse.SUPPRESS
        )  # remove model as we restore from checkpoint
        parser.add_argument("--name", default="", help=argparse.SUPPRESS)

        parser.set_defaults(**defaults)

    @staticmethod
    def _validate(opt):
        super(TestOptions, TestOptions)._validate(opt)

        if opt.warp_checkpoint and not opt.body_dir:
            raise ValueError("Warp stage must have body_dir")
        if opt.texture_checkpoint and not opt.texture_dir:
            raise ValueError("Texture stage must have texture_dir")

        if not opt.warp_checkpoint and not opt.texture_checkpoint:
            raise ValueError("Must set either warp_checkpoint or texture_checkpoint")
